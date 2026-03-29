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
