import random
import pandas as pd
import os
from tqdm import tqdm
import argparse
import json
import logging
import numpy as np

from src import *

# ── Seeding ───────────────────────────────────────────────────────────────────
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

# ── Constants ─────────────────────────────────────────────────────────────────



# ── Helpers ───────────────────────────────────────────────────────────────────
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


def identify_stressors(data: pd.DataFrame, save_path: str, logger) -> pd.DataFrame:
    """Return data with stressor column, loading from cache if available."""
    if os.path.exists(save_path):
        logger.info("Stressor cache found — skipping identification.")
        return pd.read_json(save_path)

    logger.info("Starting stressor identification...")
    min_turns, hist_len = 5, 5

    for i in tqdm(range(len(data)), desc="Stressors"):
        if data["num_turns"].iloc[i] < min_turns:
            data.at[i, "stressor"] = "Not enough dialogue history"
        else:
            latest  = data["dialogue"].iloc[i][-hist_len:]
            history = format_dialogue_history(latest)
            target  = data["dialogue"].iloc[i][-1]["text"]
            data.at[i, "stressor"] = generate_stressor(history, target)

    data.to_json(save_path, orient="records", indent=4)
    logger.info(f"Stressor identification done. Sample: {data['stressor'].iloc[0]}")
    return data


def generate_synthetic_samples(data: pd.DataFrame, label_counts, data_size: int, logger, SAMPLE_NUM_CLASS: int) -> list:
    """Generate synthetic utterances for under-represented classes."""
    logger.info("Starting synthetic data generation...")
    synthetic = []

    for level, info in MECHANISMS.items():
        num_needed = SAMPLE_NUM_CLASS - label_counts.get(level, 0)
        if num_needed <= 0:
            continue

        num_needed = min(len(data[data["label"] == level]) * data_size, SAMPLE_NUM_CLASS)

        for i in tqdm(range(num_needed),
                      desc=f"Level {level} — {info['mechanism_name']} ({num_needed} samples)"):
            inference = get_inference_input(data, level=level, n=3)
            history   = format_dialogue_history(inference.iloc[0]["dialogue"][-5:])
            utterance = generate_synthetic(
                mechanism_name=info["mechanism_name"],
                level=level,
                definition=info["definition"],
                pattern_description=info["mechanisms"],
                example_1=inference.iloc[0]["current_text"],
                example_2=inference.iloc[1]["current_text"],
                example_3=inference.iloc[2]["current_text"],
                history=history,
                stressor=inference.iloc[0]["stressor"],
            )
            synthetic.append({
                "id":           f"train_{len(data) + len(synthetic) + 1}",
                "dialogue_id":  f"synthetic_{level}_{i}",
                "dialogue":     history + f"\nSeeker: {utterance}",
                "current_text": utterance,
                "label":        level,
                "num_turns":    len(history.split("\n")) + 1,
                "stressor":     inference.iloc[0]["stressor"],
                "is_synthetic": True,
            })

    logger.info(f"Generation done. Total synthetic samples: {len(synthetic)}")
    logger.info(f"Sample utterance: {synthetic[0]['current_text']}")
    return synthetic


def evaluate_quality(synthetic: list, logger):
    """Run quality report for each class in synthetic data."""
    logger.info("Starting quality evaluation...")
    for level, info in MECHANISMS.items():
        synth_texts = [e["current_text"] for e in synthetic if e["label"] == level]
        stressors   = [e["stressor"]      for e in synthetic if e["label"] == level]
        logger.info(f"\nEvaluating Class {level} — {info['mechanism_name']}")
        quality_report(class_label=level, synthetic_sentences=synth_texts, stressors=stressors)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Data Augmentation for Clinical Stressor Identification")
    parser.add_argument("--data-dir", type=str, default="PSYDEFCONV/input_data")
    parser.add_argument("--save-dir", type=str, default="output")
    parser.add_argument("--size",     type=int, default=500)
    args = parser.parse_args()

    SAMPLE_NUM_CLASS = args.size

    os.makedirs(args.save_dir, exist_ok=True)
    logger  = setup_logger(os.path.join(os.getcwd(), "logs"), args.size)
    json_dir = os.path.join(args.save_dir, "json")
    os.makedirs(json_dir, exist_ok=True)

    data = pd.read_json(os.path.join(args.data_dir, "train.json"))
    label_counts = data["label"].value_counts()
    data["num_turns"] = data["dialogue"].apply(lambda x: sum(len(t) for t in x))

    # Stressor identification (cached)
    data = identify_stressors(
        data,
        save_path=os.path.join(json_dir, "train_data_with_stressor.json"),
        logger=logger,
    )

    # Synthetic generation
    # check if output/json/augmented_train_data_{args.size}.json exists
    if os.path.exists(os.path.join(json_dir, f"augmented_train_data_{args.size}.json")):
        logger.info("Augmented data found — skipping generation.")
        return

    # otherwise, generate synthetic samples and save results
    synthetic = generate_synthetic_samples(data, label_counts, args.size, logger, SAMPLE_NUM_CLASS)

    with open(os.path.join(json_dir, f"synthetic_data_{args.size}.json"), "w") as f:
        json.dump(synthetic, f, indent=4)

    # Merge and save augmented dataset
    data["dialogue"] = data["dialogue"].apply(format_dialogue_history)
    augmented = pd.concat([data, pd.DataFrame(synthetic)], ignore_index=True)
    augmented.to_json(os.path.join(json_dir, f"augmented_train_data_{args.size}.json"),
                      orient="records", indent=4)

    evaluate_quality(synthetic, logger)


if __name__ == "__main__":
    main()