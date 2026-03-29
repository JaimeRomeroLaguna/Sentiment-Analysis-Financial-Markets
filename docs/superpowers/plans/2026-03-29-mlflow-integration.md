# MLflow Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the financial sentiment analysis project into a proper MLflow-instrumented Python package with experiment tracking, model registry, and reproducible `MLproject` entry points.

**Architecture:** Logic is extracted from Jupyter notebooks into focused `src/` modules. Each approach gets a CLI `evaluate.py` that logs params, metrics, artifacts, and registers a model via MLflow. `MLproject` exposes both as named entry points for one-command reproducibility.

**Tech Stack:** Python 3.10, MLflow 2.x, HuggingFace Transformers, PyTorch, scikit-learn, NLTK, pandas, openpyxl, seaborn, matplotlib, pytest

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `.gitignore` | Create | Ignore `mlruns/`, `__pycache__/`, `*.pyc`, `data/*.csv`, `data/*.xlsx` |
| `conda.yaml` | Create | Pinned reproducible environment for `mlflow run` |
| `MLproject` | Create | Named entry points: `finbert`, `dictionary` |
| `conftest.py` | Create | Add project root to `sys.path` for pytest |
| `src/__init__.py` | Create | Package marker |
| `src/utils/__init__.py` | Create | Package marker |
| `src/utils/metrics.py` | Create | `compute_metrics`, `save_confusion_matrix`, `save_classification_report` |
| `src/dictionary/__init__.py` | Create | Package marker |
| `src/dictionary/scorer.py` | Create | `load_dictionary`, `preprocess`, `score`, `predict` |
| `src/dictionary/evaluate.py` | Create | CLI entry point: load data → score → log to MLflow |
| `src/finbert/__init__.py` | Create | Package marker |
| `src/finbert/model.py` | Create | `load_pipeline`, `predict` |
| `src/finbert/evaluate.py` | Create | CLI entry point: load data → infer → log to MLflow |
| `tests/test_metrics.py` | Create | Unit tests for `utils/metrics.py` |
| `tests/test_scorer.py` | Create | Unit tests for `dictionary/scorer.py` |
| `tests/test_finbert_model.py` | Create | Unit tests for `finbert/model.py` (mocked) |
| `data/` | Create dir | Place `data.csv`, `all-data.csv`, `Loughran-McDonald_MasterDictionary_1993-2024.xlsx` here |
| `notebooks/FinbertTest2.ipynb` | Move + edit | Remove Colab imports, call `src/finbert` |
| `notebooks/DictionaryStock.ipynb` | Move + edit | Remove Colab imports, call `src/dictionary` |

---

## Task 1: Project Scaffold

**Files:**
- Create: `src/__init__.py`, `src/utils/__init__.py`, `src/finbert/__init__.py`, `src/dictionary/__init__.py`
- Create: `conftest.py`
- Create: `.gitignore`
- Create: `data/` directory with `.gitkeep`
- Move: `FinbertTest2.ipynb` → `notebooks/FinbertTest2.ipynb`
- Move: `DictionaryStock.ipynb` → `notebooks/DictionaryStock.ipynb`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p src/utils src/finbert src/dictionary tests notebooks data
touch src/__init__.py src/utils/__init__.py src/finbert/__init__.py src/dictionary/__init__.py
touch data/.gitkeep
```

- [ ] **Step 2: Create `conftest.py` at project root**

```python
# conftest.py
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
```

- [ ] **Step 3: Create `.gitignore`**

```
mlruns/
*.pyc
__pycache__/
.ipynb_checkpoints/
data/*.csv
data/*.xlsx
```

- [ ] **Step 4: Move notebooks**

```bash
mv FinbertTest2.ipynb notebooks/FinbertTest2.ipynb
mv DictionaryStock.ipynb notebooks/DictionaryStock.ipynb
```

- [ ] **Step 5: Copy data files into `data/`**

Place the following files in `data/` (not committed — listed in .gitignore):
- `data/data.csv`
- `data/all-data.csv`
- `data/Loughran-McDonald_MasterDictionary_1993-2024.xlsx`

- [ ] **Step 6: Commit**

```bash
git add .gitignore conftest.py src/ tests/ notebooks/ data/.gitkeep
git commit -m "scaffold: add src structure, move notebooks, add .gitignore"
```

---

## Task 2: `conda.yaml` + `MLproject`

**Files:**
- Create: `conda.yaml`
- Create: `MLproject`

- [ ] **Step 1: Create `conda.yaml`**

```yaml
name: financial-sentiment
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - pip:
    - mlflow>=2.10
    - transformers>=4.35
    - torch>=2.0
    - nltk>=3.8
    - pandas>=2.0
    - openpyxl>=3.1
    - scikit-learn>=1.3
    - matplotlib>=3.7
    - seaborn>=0.13
    - pytest>=7.4
```

- [ ] **Step 2: Create `MLproject`**

```yaml
name: financial-sentiment

conda_env: conda.yaml

entry_points:
  finbert:
    parameters:
      dataset: {type: str, default: "data/data.csv"}
      max_length: {type: int, default: 512}
    command: "python src/finbert/evaluate.py --dataset {dataset} --max_length {max_length}"

  dictionary:
    parameters:
      dataset: {type: str, default: "data/data.csv"}
      threshold: {type: float, default: 0.3}
    command: "python src/dictionary/evaluate.py --dataset {dataset} --threshold {threshold}"
```

- [ ] **Step 3: Commit**

```bash
git add conda.yaml MLproject
git commit -m "feat: add conda.yaml and MLproject entry points"
```

---

## Task 3: `src/utils/metrics.py` (TDD)

**Files:**
- Create: `tests/test_metrics.py`
- Create: `src/utils/metrics.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_metrics.py`:

```python
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
```

- [ ] **Step 2: Run tests and confirm they fail**

```bash
pytest tests/test_metrics.py -v
```

Expected: `ImportError` — `src.utils.metrics` does not exist yet.

- [ ] **Step 3: Implement `src/utils/metrics.py`**

```python
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
```

- [ ] **Step 4: Run tests and confirm they pass**

```bash
pytest tests/test_metrics.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/metrics.py tests/test_metrics.py
git commit -m "feat: add utils/metrics with compute_metrics, confusion matrix, report"
```

---

## Task 4: `src/dictionary/scorer.py` (TDD)

**Files:**
- Create: `tests/test_scorer.py`
- Create: `src/dictionary/scorer.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_scorer.py`:

```python
import pytest
from src.dictionary.scorer import preprocess, score, predict


def test_preprocess_lowercases_tokens():
    tokens = preprocess("Apple is GREAT")
    assert all(t == t.lower() for t in tokens)


def test_preprocess_removes_stopwords():
    tokens = preprocess("the company is great")
    assert "the" not in tokens
    assert "is" not in tokens


def test_preprocess_returns_list():
    result = preprocess("company reported earnings")
    assert isinstance(result, list)


def test_score_positive():
    pos_dict = {"gain", "profit", "growth"}
    neg_dict = {"loss", "decline"}
    tokens = ["gain", "profit", "growth"]
    assert score(tokens, pos_dict, neg_dict, threshold=0.3) == "positive"


def test_score_negative():
    pos_dict = {"gain", "profit"}
    neg_dict = {"loss", "decline", "bankrupt"}
    tokens = ["loss", "decline", "bankrupt"]
    assert score(tokens, pos_dict, neg_dict, threshold=0.3) == "negative"


def test_score_neutral_no_matches():
    pos_dict = {"gain"}
    neg_dict = {"loss"}
    tokens = ["company", "reported", "results"]
    assert score(tokens, pos_dict, neg_dict, threshold=0.3) == "neutral"


def test_score_neutral_balanced():
    pos_dict = {"gain"}
    neg_dict = {"loss"}
    # ratio = (1-1)/(1+1) = 0, within threshold
    tokens = ["gain", "loss"]
    assert score(tokens, pos_dict, neg_dict, threshold=0.3) == "neutral"


def test_predict_returns_list_same_length():
    pos_dict = {"gain"}
    neg_dict = {"loss"}
    texts = ["the gain was huge", "a massive loss occurred", "results were mixed"]
    results = predict(texts, pos_dict, neg_dict, threshold=0.3)
    assert len(results) == 3
    assert all(r in {"positive", "negative", "neutral"} for r in results)
```

- [ ] **Step 2: Run tests and confirm they fail**

```bash
pytest tests/test_scorer.py -v
```

Expected: `ImportError` — `src.dictionary.scorer` does not exist yet.

- [ ] **Step 3: Implement `src/dictionary/scorer.py`**

```python
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

_stop_words = set(stopwords.words("english"))


def load_dictionary(dict_path):
    """Load Loughran-McDonald Excel dictionary.

    Returns:
        (pos_dict, neg_dict): two sets of lowercase words.
    """
    df = pd.read_excel(dict_path)
    pos_dict = set(df[df["Positive"] != 0]["Word"].str.lower().tolist())
    neg_dict = set(df[df["Negative"] != 0]["Word"].str.lower().tolist())
    return pos_dict, neg_dict


def preprocess(text):
    """Tokenize, lowercase, and remove stopwords. Returns list of tokens."""
    tokens = word_tokenize(text)
    return [t.lower() for t in tokens if t.lower() not in _stop_words]


def score(tokens, pos_dict, neg_dict, threshold=0.3):
    """Score a token list against the dictionary.

    Returns 'positive', 'negative', or 'neutral'.
    """
    pos_count = sum(1 for t in tokens if t in pos_dict)
    neg_count = sum(1 for t in tokens if t in neg_dict)
    if pos_count + neg_count == 0:
        ratio = 0.0
    else:
        ratio = (pos_count - neg_count) / (pos_count + neg_count)
    if ratio > threshold:
        return "positive"
    elif ratio < -threshold:
        return "negative"
    return "neutral"


def predict(texts, pos_dict, neg_dict, threshold=0.3):
    """Predict sentiment labels for a list of texts."""
    return [score(preprocess(t), pos_dict, neg_dict, threshold) for t in texts]
```

- [ ] **Step 4: Run tests and confirm they pass**

```bash
pytest tests/test_scorer.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dictionary/scorer.py tests/test_scorer.py
git commit -m "feat: add dictionary scorer with preprocessing and sentiment scoring"
```

---

## Task 5: `src/dictionary/evaluate.py`

**Files:**
- Create: `src/dictionary/evaluate.py`

- [ ] **Step 1: Create `src/dictionary/evaluate.py`**

```python
import argparse
import os
import tempfile

import mlflow
import mlflow.pyfunc
import pandas as pd

from src.dictionary.scorer import load_dictionary, predict
from src.utils.metrics import compute_metrics, save_confusion_matrix, save_classification_report

DICT_PATH = "data/Loughran-McDonald_MasterDictionary_1993-2024.xlsx"
EXPERIMENT_NAME = "financial-sentiment"


class DictionaryModel(mlflow.pyfunc.PythonModel):
    """MLflow PythonModel wrapper for the dictionary-based scorer."""

    def __init__(self, pos_dict, neg_dict, threshold):
        self.pos_dict = pos_dict
        self.neg_dict = neg_dict
        self.threshold = threshold

    def predict(self, context, model_input):
        from src.dictionary.scorer import score, preprocess
        return [
            score(preprocess(t), self.pos_dict, self.neg_dict, self.threshold)
            for t in model_input["text"].tolist()
        ]


def main(dataset, threshold):
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv(dataset)
    df = df.rename(columns={"Sentence": "text", "Sentiment": "sent"})

    pos_dict, neg_dict = load_dictionary(DICT_PATH)
    df["pred"] = predict(df["text"].tolist(), pos_dict, neg_dict, threshold=threshold)

    metrics = compute_metrics(df["sent"].tolist(), df["pred"].tolist())

    with mlflow.start_run():
        mlflow.log_params({
            "approach": "dictionary",
            "dataset": os.path.basename(dataset),
            "model_name": "Loughran-McDonald",
            "threshold": threshold,
        })
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as tmp:
            cm_path = os.path.join(tmp, "confusion_matrix.png")
            report_path = os.path.join(tmp, "classification_report.json")
            save_confusion_matrix(df["sent"].tolist(), df["pred"].tolist(), cm_path)
            save_classification_report(df["sent"].tolist(), df["pred"].tolist(), report_path)
            mlflow.log_artifact(cm_path)
            mlflow.log_artifact(report_path)

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=DictionaryModel(pos_dict, neg_dict, threshold),
            code_path=["src"],
            registered_model_name="dictionary-sentiment",
        )

        run_id = mlflow.active_run().info.run_id

    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Run ID: {run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dictionary-based financial sentiment evaluation")
    parser.add_argument("--dataset", default="data/data.csv", help="Path to CSV with Sentence/Sentiment columns")
    parser.add_argument("--threshold", type=float, default=0.3, help="Sentiment ratio threshold")
    args = parser.parse_args()
    main(args.dataset, args.threshold)
```

- [ ] **Step 2: Run dictionary evaluation**

```bash
python src/dictionary/evaluate.py --dataset data/data.csv --threshold 0.3
```

Expected output:
```
Accuracy: XX.XX%
Run ID: <some-uuid>
```

A new run should appear in `mlruns/`.

- [ ] **Step 3: Run on second dataset**

```bash
python src/dictionary/evaluate.py --dataset data/all-data.csv --threshold 0.3
```

Expected: second run logged under the same `financial-sentiment` experiment.

- [ ] **Step 4: Commit**

```bash
git add src/dictionary/evaluate.py
git commit -m "feat: add dictionary evaluate.py with MLflow tracking and model registry"
```

---

## Task 6: `src/finbert/model.py` (TDD)

**Files:**
- Create: `tests/test_finbert_model.py`
- Create: `src/finbert/model.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_finbert_model.py`:

```python
from unittest.mock import MagicMock
from src.finbert.model import predict


def test_predict_returns_lowercase_label():
    mock_pipe = MagicMock()
    mock_pipe.return_value = [{"label": "POSITIVE", "score": 0.9}]
    result = predict(["The company reported strong earnings"], mock_pipe)
    assert result == ["positive"]


def test_predict_multiple_texts():
    mock_pipe = MagicMock()
    mock_pipe.side_effect = [
        [{"label": "positive", "score": 0.9}],
        [{"label": "negative", "score": 0.8}],
    ]
    result = predict(["good news", "bad news"], mock_pipe)
    assert result == ["positive", "negative"]


def test_predict_neutral_label():
    mock_pipe = MagicMock()
    mock_pipe.return_value = [{"label": "neutral", "score": 0.6}]
    result = predict(["quarterly results were released"], mock_pipe)
    assert result == ["neutral"]


def test_predict_returns_list():
    mock_pipe = MagicMock()
    mock_pipe.return_value = [{"label": "positive", "score": 0.7}]
    result = predict(["text"], mock_pipe)
    assert isinstance(result, list)
```

- [ ] **Step 2: Run tests and confirm they fail**

```bash
pytest tests/test_finbert_model.py -v
```

Expected: `ImportError` — `src.finbert.model` does not exist yet.

- [ ] **Step 3: Implement `src/finbert/model.py`**

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


def load_pipeline(model_name="ProsusAI/finbert"):
    """Load FinBERT text-classification pipeline from HuggingFace."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("text-classification", model=model, tokenizer=tokenizer, framework="pt")


def predict(texts, pipe):
    """Run FinBERT inference on a list of texts.

    Args:
        texts: list of strings
        pipe: HuggingFace text-classification pipeline

    Returns:
        list of lowercase sentiment labels: 'positive', 'negative', or 'neutral'
    """
    results = []
    for text in texts:
        out = pipe(text, truncation=True, max_length=512)
        results.append(out[0]["label"].lower())
    return results
```

- [ ] **Step 4: Run tests and confirm they pass**

```bash
pytest tests/test_finbert_model.py -v
```

Expected: all 4 tests PASS (no model download needed — all mocked).

- [ ] **Step 5: Commit**

```bash
git add src/finbert/model.py tests/test_finbert_model.py
git commit -m "feat: add finbert model.py with load_pipeline and predict"
```

---

## Task 7: `src/finbert/evaluate.py`

**Files:**
- Create: `src/finbert/evaluate.py`

> Note: Running this script downloads ~440MB of FinBERT weights from HuggingFace on first run.

- [ ] **Step 1: Create `src/finbert/evaluate.py`**

```python
import argparse
import os
import tempfile

import mlflow
import mlflow.transformers
import pandas as pd

from src.finbert.model import load_pipeline, predict
from src.utils.metrics import compute_metrics, save_confusion_matrix, save_classification_report

EXPERIMENT_NAME = "financial-sentiment"


def main(dataset, max_length):
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv(dataset)
    df = df.rename(columns={"Sentence": "text", "Sentiment": "sent"})

    print("Loading FinBERT model...")
    pipe = load_pipeline()

    print(f"Running inference on {len(df)} samples...")
    df["pred"] = predict(df["text"].tolist(), pipe)

    metrics = compute_metrics(df["sent"].tolist(), df["pred"].tolist())

    with mlflow.start_run():
        mlflow.log_params({
            "approach": "finbert",
            "dataset": os.path.basename(dataset),
            "model_name": "ProsusAI/finbert",
            "max_length": max_length,
        })
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as tmp:
            cm_path = os.path.join(tmp, "confusion_matrix.png")
            report_path = os.path.join(tmp, "classification_report.json")
            save_confusion_matrix(df["sent"].tolist(), df["pred"].tolist(), cm_path)
            save_classification_report(df["sent"].tolist(), df["pred"].tolist(), report_path)
            mlflow.log_artifact(cm_path)
            mlflow.log_artifact(report_path)

        mlflow.transformers.log_model(
            transformers_model=pipe,
            artifact_path="model",
            task="text-classification",
            registered_model_name="finbert-sentiment",
        )

        run_id = mlflow.active_run().info.run_id

    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Run ID: {run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinBERT financial sentiment evaluation")
    parser.add_argument("--dataset", default="data/data.csv", help="Path to CSV with Sentence/Sentiment columns")
    parser.add_argument("--max_length", type=int, default=512, help="Max token length for truncation")
    args = parser.parse_args()
    main(args.dataset, args.max_length)
```

- [ ] **Step 2: Run FinBERT evaluation**

```bash
python src/finbert/evaluate.py --dataset data/data.csv
```

Expected output (after model download):
```
Loading FinBERT model...
Running inference on N samples...
Accuracy: XX.XX%
Run ID: <some-uuid>
```

- [ ] **Step 3: Commit**

```bash
git add src/finbert/evaluate.py
git commit -m "feat: add finbert evaluate.py with MLflow tracking and model registry"
```

---

## Task 8: Verify MLflow UI

**Files:** none — verification only.

- [ ] **Step 1: Run all tests**

```bash
pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 2: Launch MLflow UI**

```bash
mlflow ui
```

Then open `http://localhost:5000` in a browser.

- [ ] **Step 3: Verify experiment view**

In the MLflow UI, under experiment `financial-sentiment`, confirm:
- Multiple runs appear (dictionary on data.csv, dictionary on all-data.csv, finbert on data.csv)
- Each run shows params: `approach`, `dataset`, `model_name`, `threshold` or `max_length`
- Each run shows metrics: `accuracy`, `f1_macro`, `f1_positive`, `f1_negative`, `f1_neutral`
- Each run has artifacts: `confusion_matrix.png`, `classification_report.json`, `model/`

- [ ] **Step 4: Verify Model Registry**

In the MLflow UI → Models tab, confirm:
- `finbert-sentiment` is registered with at least 1 version
- `dictionary-sentiment` is registered with at least 2 versions (one per dataset)

- [ ] **Step 5: Verify `mlflow run` entry point**

```bash
mlflow run . -e dictionary --env-manager=local
```

Expected: a new run is logged with default params (`data/data.csv`, threshold 0.3).

> Use `--env-manager=local` to skip conda env creation during development. For full reproducibility demo, omit the flag (requires conda installed).

---

## Task 9: Simplify Notebooks

**Files:**
- Modify: `notebooks/FinbertTest2.ipynb`
- Modify: `notebooks/DictionaryStock.ipynb`

- [ ] **Step 1: Update `notebooks/FinbertTest2.ipynb`**

Replace all notebook cells with the following. Open the notebook in Jupyter and replace the content cell by cell:

**Cell 1 — Setup:**
```python
import sys
sys.path.insert(0, "..")  # project root

import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.finbert.model import load_pipeline, predict
from src.utils.metrics import compute_metrics
```

**Cell 2 — Run evaluation (logs to MLflow):**
```python
# This runs evaluation and logs to MLflow automatically
# Results will appear in mlflow ui at http://localhost:5000
import subprocess
result = subprocess.run(
    ["python", "../src/finbert/evaluate.py", "--dataset", "../data/data.csv"],
    capture_output=True, text=True
)
print(result.stdout)
print(result.stderr)
```

**Cell 3 — Load results from MLflow and visualise:**
```python
client = mlflow.MlflowClient()
experiment = client.get_experiment_by_name("financial-sentiment")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="params.approach = 'finbert'",
    order_by=["start_time DESC"],
    max_results=1,
)
run = runs[0]

print("Params:", run.data.params)
print("Metrics:", run.data.metrics)
```

- [ ] **Step 2: Update `notebooks/DictionaryStock.ipynb`**

Replace all notebook cells with the following:

**Cell 1 — Setup:**
```python
import sys
sys.path.insert(0, "..")

import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.dictionary.scorer import load_dictionary, predict
from src.utils.metrics import compute_metrics
```

**Cell 2 — Run evaluation (logs to MLflow):**
```python
import subprocess
result = subprocess.run(
    ["python", "../src/dictionary/evaluate.py", "--dataset", "../data/data.csv", "--threshold", "0.3"],
    capture_output=True, text=True
)
print(result.stdout)
print(result.stderr)
```

**Cell 3 — Load results from MLflow and visualise:**
```python
client = mlflow.MlflowClient()
experiment = client.get_experiment_by_name("financial-sentiment")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="params.approach = 'dictionary'",
    order_by=["start_time DESC"],
    max_results=1,
)
run = runs[0]

print("Params:", run.data.params)
print("Metrics:", run.data.metrics)
```

**Cell 4 — Compare both approaches:**
```python
all_runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"],
)

comparison = pd.DataFrame([
    {
        "approach": r.data.params.get("approach"),
        "dataset": r.data.params.get("dataset"),
        "accuracy": r.data.metrics.get("accuracy"),
        "f1_macro": r.data.metrics.get("f1_macro"),
    }
    for r in all_runs
])

print(comparison.to_string(index=False))
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/
git commit -m "feat: simplify notebooks to use src modules and MLflow client"
```

- [ ] **Step 4: Push**

```bash
git push
```

---

## Running the Full Demo

After completing all tasks, the full reproducible demo is:

```bash
# Run both approaches
python src/dictionary/evaluate.py --dataset data/data.csv
python src/dictionary/evaluate.py --dataset data/all-data.csv
python src/finbert/evaluate.py --dataset data/data.csv

# Or via MLproject (requires conda)
mlflow run . -e dictionary
mlflow run . -e finbert

# View results
mlflow ui
# → open http://localhost:5000
```
