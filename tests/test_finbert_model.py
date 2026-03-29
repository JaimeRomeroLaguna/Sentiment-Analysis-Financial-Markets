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
