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
    parser.add_argument("--dataset", default="data/data.csv")
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()
    main(args.dataset, args.max_length)
