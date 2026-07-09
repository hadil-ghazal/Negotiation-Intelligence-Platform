# No AI was used to generate this code , authored by Hadil Ghazal on 7/8/26

# Evaluation script, to derive statistic and comparison metrics across all 3 approaches
# comparing 
#   1. Naive majority class baseline
#   2. Classical ML model: TF-IDF and Logistic Regression
#   3. Deep learning model - DistilBERT
# Outputs are saved to data/outputs for use in the final report 



from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# -------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = DATA_DIR / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

X_TEST_TFIDF_PATH = PROCESSED_DATA_DIR / "X_test_tfidf.npz"
Y_TEST_PATH = PROCESSED_DATA_DIR / "y_test.pkl"
TEST_METADATA_PATH = PROCESSED_DATA_DIR / "test_metadata.csv"

BASELINE_MODEL_PATH = MODELS_DIR / "baseline_model.pkl"
LOGISTIC_MODEL_PATH = MODELS_DIR / "logistic_regression_model.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
DISTILBERT_MODEL_DIR = MODELS_DIR / "distilbert_concord"

METRICS_OUTPUT_PATH = OUTPUTS_DIR / "evaluation_metrics.csv"
REPORT_OUTPUT_PATH = OUTPUTS_DIR / "classification_reports.txt"
PREDICTIONS_OUTPUT_PATH = OUTPUTS_DIR / "test_predictions.csv"


# -------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def load_pickle(file_path: Path):
    """Load a pickle object from disk."""
    with open(file_path, "rb") as file:
        return pickle.load(file)


def save_text(text: str, file_path: Path) -> None:
    """Save text output to disk."""
    with open(file_path, "w") as file:
        file.write(text)


def evaluate_predictions(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
    confidence_scores: np.ndarray | None = None,
) -> dict:
    """
    Compute standard classification metrics.

    Weighted precision, recall, and F1 are used because the dataset labels are
    not perfectly balanced. This prevents the majority class from dominating the
    evaluation story.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "weighted_precision": precision,
        "weighted_recall": recall,
        "weighted_f1": f1,
        "average_confidence": (
            float(np.mean(confidence_scores)) if confidence_scores is not None else np.nan
        ),
    }

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)
    print(f"Accuracy:           {metrics['accuracy']:.4f}")
    print(f"Weighted Precision: {metrics['weighted_precision']:.4f}")
    print(f"Weighted Recall:    {metrics['weighted_recall']:.4f}")
    print(f"Weighted F1:        {metrics['weighted_f1']:.4f}")

    if confidence_scores is not None:
        print(f"Average Confidence: {metrics['average_confidence']:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=label_names, zero_division=0))

    return metrics


def predict_distilbert(texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Running DistilBERT inference on test texts, the model returns logits
    + Converting logits to probabilities using softmax, then using highest probability as both the predicted class and the confidenc
    """
    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_MODEL_DIR)

    model.eval()

    predictions = []
    confidence_scores = []

    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=1)
        predicted_label = torch.argmax(probabilities, dim=1).item()
        confidence = torch.max(probabilities).item()

        predictions.append(predicted_label)
        confidence_scores.append(confidence)

    return np.array(predictions), np.array(confidence_scores)


def main() -> None:
    """Evaluating all the trained models"""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    y_test = load_pickle(Y_TEST_PATH)
    label_encoder = load_pickle(LABEL_ENCODER_PATH)
    label_names = list(label_encoder.classes_)

    test_df = pd.read_csv(TEST_METADATA_PATH)
    X_test_tfidf = sparse.load_npz(X_TEST_TFIDF_PATH)

    all_metrics = []
    report_sections = []
# -------------------------------------------------------
    # 1. Naive baseline
    # ------------------------------------------------------------------

    baseline_model = load_pickle(BASELINE_MODEL_PATH)
    baseline_predictions = baseline_model.predict([[0]] * len(y_test))

    baseline_metrics = evaluate_predictions(
        model_name="Naive Baseline",
        y_true=y_test,
        y_pred=baseline_predictions,
        label_names=label_names,
    )

    all_metrics.append(baseline_metrics)

    report_sections.append(
        "naive Baseline\n"
        + classification_report(
            y_test,
            baseline_predictions,
            target_names=label_names,
            zero_division=0,
        )
    )

    # -------------------------------------------------------
    # 2.Classical ML - TF-IDF + Logistic Regression
    # ------------------------------------------------------------------

    logistic_model = load_pickle(LOGISTIC_MODEL_PATH)
    logistic_predictions = logistic_model.predict(X_test_tfidf)
    logistic_probabilities = logistic_model.predict_proba(X_test_tfidf)
    logistic_confidence = np.max(logistic_probabilities, axis=1)

    logistic_metrics = evaluate_predictions(
        model_name="TF-IDF & Logistic Regression",
        y_true=y_test,
        y_pred=logistic_predictions,
        label_names=label_names,
        confidence_scores=logistic_confidence,
    )

    all_metrics.append(logistic_metrics)

    report_sections.append(
        "\n\nTF-IDF & Logistic Regression\n"
        + classification_report(
            y_test,
            logistic_predictions,
            target_names=label_names,
            zero_division=0,
        )
    )

    # -------------------------------------------------------
    # 3.Deep learning- DistilBERT
    # ------------------------------------------------------------------

    distilbert_predictions, distilbert_confidence = predict_distilbert(
        test_df["clean_text"].astype(str).tolist()
    )

    distilbert_metrics = evaluate_predictions(
        model_name="DistilBERT",
        y_true=y_test,
        y_pred=distilbert_predictions,
        label_names=label_names,
        confidence_scores=distilbert_confidence,
    )

    all_metrics.append(distilbert_metrics)

    report_sections.append(
        "\n\nDistilBERT\n"
        + classification_report(
            y_test,
            distilbert_predictions,
            target_names=label_names,
            zero_division=0,
        )
    )

    # Saving outputs
    # ------------------------------------------------------------------

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(METRICS_OUTPUT_PATH, index=False)

    save_text("\n".join(report_sections), REPORT_OUTPUT_PATH)

    prediction_output = test_df.copy()
    prediction_output["true_label"] = label_encoder.inverse_transform(y_test)
    prediction_output["baseline_prediction"] = label_encoder.inverse_transform(baseline_predictions)
    prediction_output["logistic_prediction"] = label_encoder.inverse_transform(logistic_predictions)
    prediction_output["distilbert_prediction"] = label_encoder.inverse_transform(distilbert_predictions)
    prediction_output["logistic_confidence"] = logistic_confidence
    prediction_output["distilbert_confidence"] = distilbert_confidence

    prediction_output.to_csv(PREDICTIONS_OUTPUT_PATH, index=False)

    print("\n" + "=" * 70)
    print("Evaluation complete.")
    print("=" * 70)
    print(f"Metrics saved to: {METRICS_OUTPUT_PATH}")
    print(f"Classification reports saved to: {REPORT_OUTPUT_PATH}")
    print(f"Predictions saved to: {PREDICTIONS_OUTPUT_PATH}")


if __name__ == "__main__":
    main()