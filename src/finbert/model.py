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
