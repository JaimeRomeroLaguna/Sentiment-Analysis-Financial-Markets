import json
import os
import tempfile
import pytest
from src.utils.metrics import compute_metrics, save_confusion_matrix, save_classification_report


def test_compute_metrics_perfect_accuracy():
    y = ["positive", "negative", "neutral"]
    result = compute_metrics(y, y)
    assert result["accuracy"] == 1.0
    assert result["f1_macro"] == 1.0
    assert result["f1_positive"] == 1.0
    assert result["f1_negative"] == 1.0
    assert result["f1_neutral"] == 1.0


def test_compute_metrics_partial():
    y_true = ["positive", "positive", "negative"]
    y_pred = ["positive", "negative", "negative"]
    result = compute_metrics(y_true, y_pred)
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert 0 < result["f1_macro"] < 1.0


def test_compute_metrics_keys():
    y = ["positive", "negative", "neutral"]
    result = compute_metrics(y, y)
    expected_keys = {
        "accuracy", "precision_macro", "recall_macro", "f1_macro",
        "f1_positive", "f1_negative", "f1_neutral",
    }
    assert expected_keys == set(result.keys())


def test_save_confusion_matrix_creates_png():
    y = ["positive", "negative", "neutral"]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cm.png")
        save_confusion_matrix(y, y, path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


def test_save_classification_report_creates_valid_json():
    y = ["positive", "negative", "neutral"]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "report.json")
        save_classification_report(y, y, path)
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert "positive" in data
        assert "macro avg" in data
