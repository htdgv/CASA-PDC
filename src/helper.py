import random
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report
from tqdm import tqdm

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="  Train", leave=False)
    for batch in pbar:
        optimizer.zero_grad()
        logits = model(
            batch['input_ids'].to(device),
            batch['attention_mask'].to(device),
            batch['meta_features'].to(device),
            batch['dmrs_features'].to(device)
        )
        loss = criterion(logits, batch['labels'].to(device))
        loss.backward()
        # Gradient clipping prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(loader)


def evaluate_model(model, dataloader, device, criterion=None):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            logits = model(
                batch['input_ids'].to(device),
                batch['attention_mask'].to(device),
                batch['meta_features'].to(device),
                batch['dmrs_features'].to(device)
            )
            if criterion is not None:
                total_loss += criterion(logits, batch['labels'].to(device)).item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch['labels'].numpy())

    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    avg_loss = total_loss / len(dataloader) if criterion is not None else None
    return all_labels, all_preds, kappa, avg_loss

def plot_training_curves(history: dict, save_path: str):
    """Plot train/val loss and kappa over epochs."""
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(epochs, history['train_loss'], 'o-', label='Train Loss')
    axes[0].plot(epochs, history['val_loss'],   's--', label='Val Loss')
    axes[0].set_title('Loss per Epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Cross-Entropy Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Kappa
    axes[1].plot(epochs, history['val_kappa'], 'D-', color='darkorange', label='Val Kappa (QWK)')
    axes[1].set_title('Quadratic Weighted Kappa per Epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Cohen Kappa (quadratic)')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, n_classes: int, save_path: str, title: str = 'Confusion Matrix'):
    """Plot and save a normalized confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, data, fmt, subtitle in zip(
        axes,
        [cm, cm_norm],
        ['d', '.2f'],
        ['Raw Counts', 'Row-Normalized']
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap='Blues',
            xticklabels=range(n_classes),
            yticklabels=range(n_classes),
            ax=ax, linewidths=0.5
        )
        ax.set_title(f'{title} — {subtitle}')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_per_class_f1(y_true, y_pred, n_classes: int, save_path: str):
    """Bar chart of per-class F1."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    f1s = [report.get(str(c), {}).get('f1-score', 0.0) for c in range(n_classes)]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(n_classes), f1s, color='steelblue', edgecolor='white')
    ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=8)
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels([f'Class {c}' for c in range(n_classes)], rotation=45, ha='right')
    ax.set_ylim(0, 1.05)
    ax.set_title('Per-class F1 Score (Test Set)')
    ax.set_ylabel('F1 Score')
    ax.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()