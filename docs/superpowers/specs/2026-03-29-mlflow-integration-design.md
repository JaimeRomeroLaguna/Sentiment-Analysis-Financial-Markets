# MLflow Integration Design

**Date:** 2026-03-29
**Project:** Sentiment Analysis for Financial Markets
**Goal:** Add full MLflow experiment tracking, model registry, and MLproject reproducibility to demonstrate MLOps skills to recruiters and colleagues.

---

## Context

The project compares two sentiment analysis approaches on financial text:
- **FinBERT** (`ProsusAI/finbert`) — transformer-based, ~75.8% accuracy
- **Loughran-McDonald Dictionary** — lexicon-based, threshold scoring

Both currently live as Jupyter notebooks. This integration refactors them into proper Python modules instrumented with MLflow, while keeping the notebooks as lightweight wrappers.

---

## Project Structure

```
Sentiment-Analysis-Financial-Markets/
├── MLproject                        # MLflow entry points + environment
├── conda.yaml                       # Reproducible environment spec
├── src/
│   ├── finbert/
│   │   ├── evaluate.py              # Entry point: load data → predict → log to MLflow
│   │   └── model.py                 # Model loading + inference logic
│   ├── dictionary/
│   │   ├── evaluate.py              # Entry point: preprocess → score → log to MLflow
│   │   └── scorer.py                # Loughran-McDonald scoring logic
│   └── utils/
│       └── metrics.py               # Shared: accuracy, classification report, confusion matrix
├── data/
│   ├── data.csv
│   ├── all-data.csv
│   └── Loughran-McDonald_MasterDictionary_1993-2024.xlsx
├── notebooks/
│   ├── FinbertTest2.ipynb           # Calls src/finbert, visualizes MLflow results
│   └── DictionaryStock.ipynb        # Calls src/dictionary, visualizes MLflow results
└── mlruns/                          # MLflow tracking data (gitignored)
```

---

## MLproject File

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

**Usage:**
```bash
mlflow run . -e finbert
mlflow run . -e dictionary -P dataset=data/all-data.csv -P threshold=0.2
mlflow ui   # open localhost:5000
```

---

## What Gets Tracked Per Run

### Parameters
| Key | FinBERT | Dictionary |
|-----|---------|------------|
| `approach` | `finbert` | `dictionary` |
| `dataset` | filename | filename |
| `model_name` | `ProsusAI/finbert` | `Loughran-McDonald` |
| `max_length` | 512 | — |
| `threshold` | — | 0.3 |

### Metrics
- `accuracy`
- `precision_macro`, `recall_macro`, `f1_macro`
- `f1_positive`, `f1_negative`, `f1_neutral`

### Artifacts
- `confusion_matrix.png` — heatmap
- `classification_report.json` — per-class precision/recall/F1
- Logged model (via `mlflow.transformers.log_model` for FinBERT, `mlflow.pyfunc.log_model` for Dictionary)

---

## Model Registry

| Registry Name | Approach |
|---------------|----------|
| `finbert-sentiment` | FinBERT |
| `dictionary-sentiment` | Loughran-McDonald |

Versions are tagged with the dataset used. Both registered under a single MLflow experiment: `financial-sentiment`.

---

## Data Flow

```
data/data.csv
     │
     ▼
src/{approach}/evaluate.py
     │
     ├── preprocess / tokenize
     ├── predict (model inference or lexicon scoring)
     ├── compute metrics (via utils/metrics.py)
     │
     └── mlflow.start_run()
             ├── log_params(...)
             ├── log_metrics(...)
             ├── log_artifact(confusion_matrix.png)
             ├── log_artifact(classification_report.json)
             └── log_model(...) → registered in Model Registry
```

---

## Environment

`conda.yaml` pins all dependencies for full reproducibility:
- `transformers`, `torch` (FinBERT)
- `nltk`, `pandas`, `openpyxl` (Dictionary)
- `mlflow`, `scikit-learn`, `matplotlib`, `seaborn`

---

## What to Gitignore

```
mlruns/
*.pyc
__pycache__/
```
