import random
import pandas as pd
import os
from tqdm import tqdm
import argparse
import numpy as np
import logging
from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.preprocessing import MinMaxScaler

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
BASE_COLS  = ["id", "dialogue_id"]
SCALE_COLS = ["feat_length", "feat_i_density", "feat_insight", "feat_intensity", 'is_synthetic']


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

def build_pipelines(token):
    """Initialise emotion pipeline and mental-RoBERTa model."""
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    kwargs = {"torch_dtype": dtype, "attn_implementation": "eager"}

    emotion_pipe = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=0 if torch.cuda.is_available() else -1,
        model_kwargs=kwargs,
    )

    model = AutoModel.from_pretrained(
        "mental/mental-roberta-base", token=token, **kwargs
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    return emotion_pipe, model


def extract_features(data: pd.DataFrame, emotion_pipe, logger) -> pd.DataFrame:
    """Add all feat_* columns to data in-place and return it."""
    data["feat_length"]    = data["current_text"].apply(get_utterance_length)
    data["feat_i_density"] = data["current_text"].apply(get_i_pronoun_density)
    data["feat_insight"]   = data["current_text"].apply(get_insight_density)
    data["feat_phatic"]    = data["current_text"].apply(get_phatic_flag)
    data["feat_mature"]    = data.apply(get_mature_flag, axis=1)
    data["feat_feeling"]   = data["current_text"].apply(get_feeling_flag)

    logger.info("Processing emotion intensity...")
    with torch.inference_mode():
        scores = [
            out["score"]
            for out in tqdm(
                emotion_pipe(data["current_text"].tolist(), batch_size=32, truncation=True),
                total=len(data),
            )
        ]
    data["feat_intensity"] = scores
    return data


def split_and_save(data: pd.DataFrame, data_dir: str, size: int, logger) -> None:
    """Split data into text / feature / label frames and save as JSON."""
    text_df = data[BASE_COLS + ["dialogue", "current_text", "stressor"]]

    feature_df = data.drop(columns=[
        "dialogue", "current_text", "stressor", "label", "feat_phatic", "feat_mature"
    ])          # keep is_synthetic

    label_df = data[BASE_COLS + ["label"]]

    for df, name in [
        (text_df,    f"text_data_{size}_qwen.json"),
        (feature_df, f"feature_data_{size}_qwen.json"),
        (label_df,   f"label_data_{size}_qwen.json"),
    ]:
        path = os.path.join(data_dir, name)
        df.to_json(path, orient="records", lines=True)
        logger.info(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",  type=str, default="output/json")
    parser.add_argument("--token",     type=str, default=None)
    parser.add_argument("--save-dir",  type=str, default="output")
    parser.add_argument("--data-size", type=int, default=1000)
    args = parser.parse_args()
    logger  = setup_logger(os.path.join(os.getcwd(), "logs"), args.data_size)

    set_seed(SEED)

    emotion_pipe, _ = build_pipelines(args.token)

    data = pd.read_json(
        os.path.join(args.data_dir, f"augmented_train_data_{args.data_size}_qwen.json")
    )
    logger.info(f"Loaded {len(data)} rows\n{data['label'].value_counts()}")

    data = extract_features(data, emotion_pipe, logger)
    data.drop(columns=["num_turns"], inplace=True, errors="ignore")

    data[SCALE_COLS] = MinMaxScaler().fit_transform(data[SCALE_COLS])

    split_and_save(data, args.data_dir, args.data_size, logger)


if __name__ == "__main__":
    main()