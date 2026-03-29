import json
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for scripts
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def compute_metrics(y_true, y_pred):
    """Return dict of accuracy, macro precision/recall/F1, and per-class F1."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": report["macro avg"]["precision"],
        "recall_macro": report["macro avg"]["recall"],
        "f1_macro": report["macro avg"]["f1-score"],
        "f1_positive": report.get("positive", {}).get("f1-score", 0.0),
        "f1_negative": report.get("negative", {}).get("f1-score", 0.0),
        "f1_neutral": report.get("neutral", {}).get("f1-score", 0.0),
    }


def save_confusion_matrix(y_true, y_pred, path):
    """Save a confusion matrix heatmap PNG to path."""
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_classification_report(y_true, y_pred, path):
    """Save classification report as JSON to path."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
