# Mitigating Data Scarcity in Psychological Defense Classification with Context-Aware Synthetic Augmentation

[![Paper](https://img.shields.io/badge/Paper-ACL%202026-blue)](https://arxiv.org/abs/2605.14380) 
[![Code](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/htdgv/CASA-PDC)

> Accepted at **BioNLP @ ACL 2026** — PsyDefDetect Shared Task

---

## Table of Contents
- [Overview](#overview)
- [Results](#results)
- [Installation](#installation)
- [Data](#data)
- [Training](#training)
- [Contact](#contact)
- [Citation](#citation)

---

## Overview

Psychological defense mechanisms (PDMs) are unconscious cognitive processes that modulate how individuals perceive and respond to emotional distress. Automatically classifying PDMs from text is clinically valuable but severely hindered by data scarcity and class imbalance — challenges that generative augmentation alone cannot resolve without psychological grounding.

We address these challenges in the **PsyDefDetect shared task (BioNLP@ACL 2026)** by proposing a **context-aware synthetic augmentation framework** combined with a hybrid classification model. Our hybrid model integrates contextual language representations with clinical features, alongside 150 annotated defense items. Experiments demonstrate that definition quality in prompting directly governs generation fidelity and downstream performance.

![Overview](asset/psydefdetect-Copy%20of%20Page-1.drawio.png)

---

## Results

Our method surpasses the DMRS Co-Pilot baseline, establishing a strong benchmark for psychologically grounded defense mechanism classification in low-resource settings.

| Model | Accuracy | Macro-F1 |
|---|---|---|
| DMRS Co-Pilot (baseline) | 18.01% | 8.63% |
| **CASA-PDC (ours)** | **58.26%** | **24.62%** |
| Δ improvement | +40.25% | +15.99% |

---

## Installation

Requires **Python 3.10** and **CUDA 12.x** for GPU support. Adjust the CUDA version in `requirements.txt` if needed.

```bash
conda create -n psydef_env python=3.10 -y
conda activate psydef_env
pip install -r requirements.txt
```

---

## Data

This project uses the **PsyDefDetect** dataset from the BioNLP@ACL 2026 shared task.

### Downloading the Data

1. Register and download the dataset from the official shared task page:
   👉 [PsyDefDetect Shared Task](https://psydefdetect-shared-task.github.io/)

2. Place the downloaded files in the `data/` directory:

```txt
data/
├── train.json
└── test.json
```

3. Run the preprocessing script to prepare the data for training:

```bash
python scripts/preprocess.py --data_dir data/ --output_dir data/processed/
```

> **Note:** The dataset is provided solely for research purposes under the terms of the shared task organizers. Please review their data usage agreement before downloading.

---

## Training

This pipeline uses [MentalRoBERTa](https://huggingface.co/mental/mental-roberta-base) as the backbone model. Access requires approval from the model owner on Hugging Face.

**Setup your Hugging Face token:**
```bash
huggingface-cli login
# or set the environment variable:
export HUGGINGFACE_TOKEN=your_token_here
```

**Run training:**
```bash
chmod +x script/run.sh
./script/run.sh
```

You will be prompted to enter your Hugging Face API key if not already set. After successful authentication, training will begin on the PsyDefDetect dataset. Outputs and checkpoints are saved to the directory specified in `script/run.sh`.

For hyperparameter details and evaluation metrics, refer to `script/run.sh` and the corresponding sections in the paper.

---
## Contact
For questions or issues, please contact: 
- **Hoang-Thuy-Duong Vu** — [26duong.vht@vinuni.edu.vn](mailto:26duong.vht@vinuni.edu.vn)
- **Dr. Huy-Hieu Pham** - [hieu.ph@vinuni.edu.vn](mailto:hieu.ph@vinuni.edu.vn)

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{vu2026mitigatingdatascarcitypsychological,
      title={Mitigating Data Scarcity in Psychological Defense Classification with Context-Aware Synthetic Augmentation}, 
      author={Hoang-Thuy-Duong Vu and Quoc-Cuong Pham and Huy-Hieu Pham},
      year={2026},
      eprint={2605.14380},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.14380}, 
}
```