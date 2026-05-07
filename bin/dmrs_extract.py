import random
import pandas as pd
import os
import json
import numpy as np
import argparse
from tqdm import tqdm
from collections import defaultdict
from transformers import pipeline
from sklearn.model_selection import train_test_split
import logging

from src import *

# --- SEEDING ---
SEED = 2025
os.environ.update({
    "PYTHONHASHSEED": str(SEED),
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
})
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)


def setup_logger(log_dir: str, size: int) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f"run_{size}.log")),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def load_data(data_dir, data_size):
    text_df = pd.read_json(os.path.join(data_dir, f"text_data_{data_size}_qwen.json"), orient="records", lines=True)
    feature_df = pd.read_json(os.path.join(data_dir, f"feature_data_{data_size}_qwen.json"), orient="records", lines=True)
    label_df = pd.read_json(os.path.join(data_dir, f"label_data_{data_size}_qwen.json"), orient="records", lines=True)
    feature_df = feature_df.drop(columns=["label"], errors="ignore")
    return text_df, feature_df, label_df


def get_train_texts(text_df, label_df):
    id_label = label_df[["id", "label"]].drop_duplicates()
    train_ids, _ = train_test_split(
        id_label["id"], test_size=0.2, stratify=id_label["label"], random_state=42
    )
    return text_df[text_df["id"].isin(set(train_ids))].reset_index(drop=True)


def build_dmrs_items():
    counts = defaultdict(int)
    dmrs_map = {}
    for k, v in ITEMS.items():
        counts[v] += 1
        dmrs_map[k] = f"{v}_{counts[v]}"
    return list(dmrs_map.values())


def run_inference(nli_model, train_text_df, dmrs_items, device):
    batch_size = 16 if device == 0 else 1
    texts = train_text_df["current_text"].tolist()
    ids = train_text_df["id"].tolist()

    results = []
    for i, result in enumerate(tqdm(
        nli_model(texts, dmrs_items, multi_label=True, batch_size=batch_size),
        total=len(texts), desc="Inference"
    )):
        item_weights = dict(zip(result["labels"], result["scores"]))
        mechanism = calculate_dmrs_mechanism(item_weights)
        results.append({"id": ids[i], **mechanism})

    return results


def save_results(results, output_path, logger):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Done! Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",  type=str, default="output/json")
    parser.add_argument("--token",     type=str, default=None)
    parser.add_argument("--data-size", type=int, default=1000)
    args = parser.parse_args()
    logger  = setup_logger(os.path.join(os.getcwd(), "logs"), args.data_size)

    set_seed(SEED)

    device = 0 if torch.cuda.is_available() else -1
    logger.info(f"PyTorch {torch.__version__} | CUDA {torch.version.cuda} | Device: {'GPU' if device == 0 else 'CPU'}")
    if device == 0:
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)} | CC: {torch.cuda.get_device_capability(0)}")

    nli_model = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device,
        torch_dtype=torch.float16 if device == 0 else torch.float32,
        token=args.token,
    )

    text_df, _, label_df = load_data(args.data_dir, args.data_size)
    train_text_df = get_train_texts(text_df, label_df)
    dmrs_items = build_dmrs_items()

    logger.info(f"Starting batched inference for {len(train_text_df)} samples...")
    results = run_inference(nli_model, train_text_df, dmrs_items, device)

    output_path = os.path.join(args.data_dir, f"dmrs_mechanism_scores_{args.data_size}_qwen.json")
    save_results(results, output_path, logger)


if __name__ == "__main__":
    main()