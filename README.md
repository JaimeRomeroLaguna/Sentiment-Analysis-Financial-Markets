# Sentiment Analysis for Financial Markets

This project explores two different approaches to sentiment analysis on financial text: a neural (transformer-based) method and a classical lexicon-based method. Both are evaluated on labeled datasets and tested on real-world financial news.

---

## Notebooks

### 1. `FinbertTest2.ipynb` — Neural Approach with FinBERT

Uses **FinBERT** (`ProsusAI/finbert`), a BERT model fine-tuned on financial text, to classify sentences as **positive**, **negative**, or **neutral**.

**Pipeline:**
1. Load a labeled financial sentences dataset (`data.csv`)
2. Load the pre-trained FinBERT model via Hugging Face Transformers
3. Run inference on each sentence (truncated to 512 tokens)
4. Compare predictions against ground truth labels
5. Evaluate with accuracy score, classification report, and confusion matrix heatmap

**Result:** ~75.8% accuracy on the test dataset.

**Requirements:** `transformers`, `torch`, `pandas`, `scikit-learn`, `seaborn`, `matplotlib`

---

### 2. `DictionaryStock.ipynb` — Lexicon-Based Approach

Uses the **Loughran-McDonald Master Dictionary** (1993–2024), a domain-specific financial lexicon, to score sentiment by counting positive and negative words in each document.

**Pipeline:**
1. Preprocess text: tokenize, lowercase, remove stopwords (NLTK)
2. Load the Loughran-McDonald dictionary from Excel and extract positive/negative word lists
3. Score each document: `(pos - neg) / (pos + neg)`
4. Classify using a threshold (`> 0.3` → positive, `< -0.3` → negative, else neutral)
5. Evaluate on two labeled datasets (`data.csv`, `all-data.csv`)
6. Fetch real-time news via **NewsAPI** for tickers (Boeing, NVIDIA, Netflix) and analyze sentiment distribution

**Result:** Evaluated on two datasets with confusion matrices and classification reports. Real-time news sentiment shown with bar and pie charts.

**Requirements:** `nltk`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `requests`, `langdetect`, `feedparser`

---

## Approach Comparison

| | FinBERT | Dictionary |
|---|---|---|
| Method | Neural (transformer) | Lexicon matching |
| Accuracy | ~75.8% | Varies by dataset |
| Speed | Slower (model inference) | Fast (word matching) |
| Interpretability | Low (black-box) | High (word-level) |
| Real-time news | No | Yes (NewsAPI) |

---

## Data

- `data.csv` — Labeled financial sentences (Kaggle: Financial Sentiment Analysis)
- `all-data.csv` — Labeled financial news headlines (Kaggle: Sentiment Analysis for Financial News)
- `Loughran-McDonald_MasterDictionary_1993-2024.xlsx` — Financial sentiment lexicon
- NewsAPI — Real-time news fetched at runtime (requires API key)
