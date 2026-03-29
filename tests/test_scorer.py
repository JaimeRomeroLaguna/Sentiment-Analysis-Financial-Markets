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
    tokens = ["gain", "loss"]
    assert score(tokens, pos_dict, neg_dict, threshold=0.3) == "neutral"


def test_predict_returns_list_same_length():
    pos_dict = {"gain"}
    neg_dict = {"loss"}
    texts = ["the gain was huge", "a massive loss occurred", "results were mixed"]
    results = predict(texts, pos_dict, neg_dict, threshold=0.3)
    assert len(results) == 3
    assert all(r in {"positive", "negative", "neutral"} for r in results)
