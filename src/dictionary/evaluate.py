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
