import pandas as pd
import os
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup, pipeline
from torch.optim import AdamW
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report
from collections import defaultdict
from src import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aug-dir',    type=str, default='output')
    parser.add_argument('--save-dir',   type=str, default='output/out')
    parser.add_argument('--size',       type=int, default=500)
    parser.add_argument('--token',      type=str, default=None)
    parser.add_argument('--epochs',     type=int, default=10)
    parser.add_argument('--lr',         type=float, default=1e-5)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--patience',   type=int, default=3,
                        help='Early stopping patience (epochs without kappa improvement)')
    parser.add_argument('--grad-clip',  type=float, default=1.0)
    parser.add_argument('--blind-test',  type=str, default='PSYDEFCONV/input_data/test_label.json')

    args = parser.parse_args()

    set_seed(2025)
    os.makedirs(args.save_dir, exist_ok=True)

    # ── Logging ──────────────────────────────
    LOG_DIR = os.path.join(os.getcwd(), "logs")
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_PATH = os.path.join(LOG_DIR, f"run_{args.size}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)

    # ── Data loading ─────────────────────────
    text_df = pd.read_json(f'{args.aug_dir}/text_data_{args.size}.json',    orient='records', lines=True)
    feat_df = pd.read_json(f'{args.aug_dir}/feature_data_{args.size}.json', orient='records', lines=True)
    lab_df  = pd.read_json(f'{args.aug_dir}/label_data_{args.size}.json',   orient='records', lines=True)
    dmrs_df = pd.read_json(f'{args.aug_dir}/dmrs_mechanism_scores_{args.size}.json')

    common_ids = sorted(set(text_df['id']) & set(feat_df['id']) & set(lab_df['id']) & set(dmrs_df['id']))
    logger.info(f"Aligned samples: {len(common_ids)}")

    text_df = text_df.set_index('id').loc[common_ids].reset_index()
    feat_df = feat_df.set_index('id').loc[common_ids].reset_index()
    lab_df  = lab_df.set_index('id').loc[common_ids].reset_index()
    dmrs_df = dmrs_df.set_index('id').loc[common_ids].reset_index()

    # ── Split ────────────────────────────────
    train_idx, test_idx = train_test_split(
        range(len(common_ids)), test_size=0.2,
        stratify=lab_df['label'], random_state=42
    )

    # ── Feature scaling ──────────────────────
    meta_cols = ['feat_length', 'feat_i_density', 'feat_insight',
                 'feat_intensity', 'feat_feeling']
    dmrs_cols = [c for c in dmrs_df.select_dtypes(include=[np.number]).columns
                 if c not in ['id', 'label']]

    s_meta, s_dmrs = StandardScaler(), StandardScaler()
    tr_meta = s_meta.fit_transform(feat_df.iloc[train_idx][meta_cols].fillna(0))
    ts_meta = s_meta.transform(feat_df.iloc[test_idx][meta_cols].fillna(0))
    tr_dmrs = s_dmrs.fit_transform(dmrs_df.iloc[train_idx][dmrs_cols].fillna(0))
    ts_dmrs = s_dmrs.transform(dmrs_df.iloc[test_idx][dmrs_cols].fillna(0))

    # ── Tokenizer & datasets ─────────────────
    MODEL_NAME = "mental/mental-roberta-base"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=args.token)

    def fmt(df):
        return ("Stressor: " + df['stressor'].astype(str)
                + " | Turn: " + df['current_text'].astype(str)).tolist()

    train_ds = PsychologicalDefenseDataset1(
        fmt(text_df.iloc[train_idx]), tr_meta, tr_dmrs,
        lab_df.iloc[train_idx]['label'].tolist(), tokenizer
    )
    test_ds = PsychologicalDefenseDataset1(
        fmt(text_df.iloc[test_idx]), ts_meta, ts_dmrs,
        lab_df.iloc[test_idx]['label'].tolist(), tokenizer
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # ── Model, loss, optimizer, scheduler ────
    n_classes = 9
    model = HybridDefenseClassifier1(
        n_classes, len(meta_cols), len(dmrs_cols), MODEL_NAME, args.token
    ).to(device)

    counts  = np.bincount(lab_df.iloc[train_idx]['label'], minlength=n_classes)
    weights = torch.tensor(1.0 / np.sqrt(counts + 1e-6), dtype=torch.float).to(device)
    weights = weights / weights.sum() * n_classes

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)          # 10 % warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # ── Training loop ────────────────────────
    best_kappa    = -1.0
    patience_left = args.patience
    history = {'train_loss': [], 'val_loss': [], 'val_kappa': []}

    for epoch in range(1, args.epochs + 1):
        logger.info(f"── Epoch {epoch}/{args.epochs} ──")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, args.grad_clip
        )
        y_t, y_p, kappa, val_loss = evaluate_model(
            model, test_loader, device, criterion=criterion
        )

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_kappa'].append(kappa)

        logger.info(
            f"  Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Kappa (QWK): {kappa:.4f}"
        )

        if kappa > best_kappa:
            best_kappa    = kappa
            patience_left = args.patience
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pt"))
            logger.info(f"  ✓ New best kappa={best_kappa:.4f} — model saved")
        else:
            patience_left -= 1
            logger.info(f"  No improvement. Patience left: {patience_left}/{args.patience}")
            if patience_left == 0:
                logger.info("Early stopping triggered.")
                break

    # ── Final evaluation ─────────────────────
    model.load_state_dict(torch.load(os.path.join(args.save_dir, "best_model.pt")))
    y_t, y_p, final_kappa, _ = evaluate_model(model, test_loader, device)

    logger.info(f"\nFinal Test QWK: {final_kappa:.4f}")
    print("\n" + classification_report(y_t, y_p, digits=4))

    # ── Plots ─────────────────────────────────
    plot_training_curves(
        history,
        save_path=os.path.join(args.save_dir, "training_curves.png")
    )
    plot_confusion_matrix(
        y_t, y_p, n_classes,
        save_path=os.path.join(args.save_dir, "confusion_matrix.png")
    )
    plot_per_class_f1(
        y_t, y_p, n_classes,
        save_path=os.path.join(args.save_dir, "per_class_f1.png")
    )
    logger.info(f"Plots saved to {args.save_dir}/")





    # BLIND TEST INFERENCE

    # Load Mental-RoBERTa
    tokenizer = AutoTokenizer.from_pretrained("mental/mental-roberta-base", token=args.token)
    model = AutoModel.from_pretrained("mental/mental-roberta-base", token=args.token)

    emotion_pipe = pipeline("sentiment-analysis", 
                            model='cardiffnlp/twitter-roberta-base-sentiment-latest',
                            tokenizer=tokenizer,
                            token=args.token,
                            device=0 if torch.cuda.is_available() else -1)

    # load train data first
    data = pd.read_json(args['blind_test'], orient='records', lines=True)

    # calculate num_turns for each dialogue
    data['num_turns'] = data['dialogue'].apply(len)

    # logger.info data general informations
    logger.info(f"Number of rows: {data.shape[0]}")
    logger.info(f"Number of unique dialogues: {data['dialogue_id'].nunique()}")
    logger.info(f"Number of data used for training: {data['id'].nunique()}")

    # Stressor identification

    min_num_turns = 5
    hist_length = 5

    for i in tqdm(range(len(data))):

        if data['num_turns'].iloc[i] < min_num_turns:
            data.at[i, 'stressor'] = "Not enough dialogue history"
        else: 
            # get 5 latest turns
            latest_turns = data['dialogue'].iloc[i][-hist_length:]
            history = format_dialogue_history(latest_turns)
            target_turn = data['dialogue'].iloc[i][-1]['text']
            stressor = generate_stressor(history, target_turn)
            data.at[i, 'stressor'] = stressor


    data['dialogue'] = data['dialogue'].apply(lambda x: format_dialogue_history(x))
        

    data['feat_length'] = data['current_text'].apply(get_utterance_length)
    data['feat_i_density'] = data['current_text'].apply(get_i_pronoun_density)
    data['feat_insight'] = data['current_text'].apply(get_insight_density)
    data['feat_intensity'] = data['current_text'].apply(get_emotion_intensity, emotion_pipe=emotion_pipe)
    data['feat_phatic'] = data['current_text'].apply(get_phatic_flag)
    data['feat_mature'] = data.apply(get_mature_flag, axis=1)
    data['feat_feeling'] = data['current_text'].apply(get_feeling_flag)

    # minmax scale feat_length, feat_i_density, feat_insight, feat_intensity
    scaler = MinMaxScaler()
    data[['feat_length', 'feat_i_density', 'feat_insight', 'feat_intensity']] = scaler.fit_transform(data[['feat_length', 'feat_i_density', 'feat_insight', 'feat_intensity']])

    # separate text data and feature data
    base_col = ['id','dialogue_id']
    test_text_df = data[base_col + [
        'dialogue',
        'current_text',
        'stressor'
    ]]

    test_feature_df = data.drop(columns=[
        'dialogue',
        'current_text',
        'stressor'
    ])

    # Define the exact features we want (7 features total)
    feature_cols = ['feat_length', 'feat_i_density', 'feat_insight', 
                    'feat_intensity', 'feat_feeling']

    scaler = StandardScaler()
    train_meta_scaled = scaler.fit_transform(feat_df.iloc[train_idx][meta_cols].fillna(0))
    test_meta_scaled = scaler.transform(test_feature_df[feature_cols])
    test_texts = (
        "Stressor: " + test_text_df['stressor'].astype(str) + 
        " | Turn: " + test_text_df['current_text'].astype(str)
    ).tolist()

    dummy_labels = [0] * len(test_texts)

    texts_to_process = test_text_df['current_text'].tolist()
    ids_to_process = test_text_df["id"].tolist()

    counts = defaultdict(int)
    DMRS_ITEMS_MAP = {}
    for k, v in ITEMS.items():
        counts[v] += 1
        DMRS_ITEMS_MAP[k] = f"{v}_{counts[v]}"
    DMRS_ITEMS = list(DMRS_ITEMS_MAP.values())

    # Initialize pipeline with GPU and FP16 (Half precision) for speed
    nli_model = pipeline(
        "zero-shot-classification", 
        model="facebook/bart-large-mnli",
        device=device,
        torch_dtype=torch.float16 if device == 0 else torch.float32,
        token=args.token
    )

    # --- BATCHED INFERENCE (THE ACCELERATED PART) ---
    feature_rows = []
    json_results = []

    batch_size = 16 if device == 0 else 1 # Adjust based on your VRAM

    print(f"Processing {len(texts_to_process)} samples with batch size {batch_size}...")
    results_gen = nli_model(texts_to_process, DMRS_ITEMS, multi_label=True, batch_size=batch_size)
    print("Inference completed. Processing results...")

    for i, result in enumerate(tqdm(results_gen, total=len(texts_to_process))):
        sample_id = ids_to_process[i]
        
        # Map labels back to scores
        item_weights = dict(zip(result['labels'], result['scores']))
        
        # Calculate mechanism scores
        sample_mechanism = calculate_dmrs_mechanism(item_weights)

        # Store results
        feature_rows.append({"id": sample_id, **sample_mechanism})
        json_results.append({
            "id": sample_id,
            "dmrs_mechanisms": sample_mechanism
        })

    test_dataset = PsychologicalDefenseDataset1(
        texts=test_texts,
        meta_features=test_meta_scaled,
        dmrs_features=np.zeros((len(test_meta_scaled), len(dmrs_cols))),  # Placeholder DMRS features
        labels=dummy_labels,
        tokenizer=tokenizer
    )

    # 1. LOAD THE BEST MODEL
    # Ensure you use the same architecture parameters as training
    model = HybridDefenseClassifier1(n_classes=9, n_meta_features=len(feature_cols), model_name=MODEL_NAME, n_dmrs_features=len(dmrs_cols), token=args.token).to(device)
    model.load_state_dict(torch.load(os.path.join(args.save_dir, "best_model.pt")))
    model.eval()

    # Use a batch_size of 1 for "one-by-one" or larger for speed
    inference_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 3. RUN INFERENCE
    all_predictions = []

    logger.info("Starting inference on unlabeled test set...")
    with torch.no_grad():
        for batch in tqdm(inference_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            meta_features = batch['meta_features'].to(device)
            dmrs_features = batch['dmrs_features'].to(device)
            # Get model outputs
            logits = model(input_ids, attention_mask, meta_features, dmrs_features)

            # Convert logits to class indices (0-8)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_predictions.extend(preds)

    # 4. SAVE THE RESULTS
    # Match the predictions back to the original IDs
    results_df = pd.DataFrame({
        'id': test_text_df['id'],
        # 'dialogue_id': test_text_df['dialogue_id'],
        'label': all_predictions
    })

    # Optional: Map the numbers back to names for easier reading
    # defense_mapping = {0: "Phatic", 1: "Mature", 2: "...", etc.}
    # results_df['defense_name'] = results_df['predicted_label'].map(defense_mapping)

    # save json
    results_df.to_json(os.path.join(args.save_dir, f"test_predictions_results_{args.size}.json"), orient='records', lines=True)
    logger.info(f"Inference complete. Results saved to {os.path.join(args.save_dir, f'test_predictions_results_{args.size}.json')}.")

    # 5. RESEARCHER'S CHECK: Distribution analysis
    logger.info("\nPredicted Class Distribution:")
    logger.info(results_df['predicted_label'].value_counts().sort_index())



    ####################### 
    # TEST DATA
    ####################### 

    test_data = pd.read_json(args['blind_test'], orient='records', lines=True)
    results_df['ground_truth'] = test_data['label'].values

    # classification report
    logger.info("\nClassification Report on Test Set:")
    logger.info(classification_report(results_df['ground_truth'], results_df['predicted_label'], digits=4))
    logger.info("\nConfusion Matrix on Test Set:")
    # plot fig of confusion matrix
    plot_confusion_matrix(
        results_df['ground_truth'], results_df['predicted_label'], n_classes,
        save_path=os.path.join(args.save_dir, "confusion_matrix_test_set.png")
    )
    logger.info(f"Confusion matrix saved to {os.path.join(args.save_dir, 'confusion_matrix_test_set.png')}.")
    logger.info("done.")



if __name__ == "__main__":
    main()