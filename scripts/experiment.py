# No AI was used to generate this code , authored by Hadil Ghazal on 7/8/26
"""Robustness experiment for the negotiation intelligence Concord:
This section tests whether the trained models remain reliable when the
negotiation transcript is degraded with simple text noise

Experiment:
- Original test text
- Mild noise
- Moderate noise
- Heavy noise

Models tested:
- TF-IDF + Logistic Regression
- DistilBERT
- The naive baseline is excluded because it ignores text entirely so the text noise can’t impact its predictions at all
"""

from pathlib import Path
import pickle
import random
import re

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = DATA_DIR / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

TEST_METADATA_PATH = PROCESSED_DATA_DIR / "test_metadata.csv"
Y_TEST_PATH = PROCESSED_DATA_DIR / "y_test.pkl"

TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
LOGISTIC_MODEL_PATH = MODELS_DIR / "logistic_regression_model.pkl"
DISTILBERT_MODEL_DIR = MODELS_DIR / "distilbert_concord"

EXPERIMENT_OUTPUT_PATH = OUTPUTS_DIR / "robustness_experiment_results.csv"

RANDOM_SEED = 42


def load_pickle(file_path: Path):
    """Load a saved Python object."""
    with open(file_path, "rb") as file:
        return pickle.load(file)


def add_text_noise(text: str, noise_level: float) -> str:
    # Here, randomly removing words from text 
    # ... to simulates real world messy negotiation input such as incomplete notes,rushed summaries, missing context, or partially copied notes and transcripts

    words = str(text).split()

    if not words:
        return text

    kept_words = [
        word for word in words
        if random.random() > noise_level
    ]

    return " ".join(kept_words) if kept_words else text


def clean_text(text: str) -> str:
   #Matching the conservative cleaning method used in build_features.py.
    
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s.,;:!?'-]", "", text)
    return text.strip()


def predict_logistic(texts: list[str]) -> np.ndarray:
    #Predicting labels using the trained TF-IDFLogistic Regression model
    vectorizer = load_pickle(TFIDF_VECTORIZER_PATH)
    model = load_pickle(LOGISTIC_MODEL_PATH)

    cleaned_texts = [clean_text(text) for text in texts]
    features = vectorizer.transform(cleaned_texts)

    return model.predict(features)


def predict_distilbert(texts: list[str]) -> np.ndarray:
    #Predicting labels using the trained DistilBERT model
    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_MODEL_DIR)

    model.eval()
    predictions = []

    for text in texts:
        inputs = tokenizer(
            str(text),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        )

        with torch.no_grad():
            outputs = model(**inputs)

        predicted_label = torch.argmax(outputs.logits, dim=1).item()
        predictions.append(predicted_label)

    return np.array(predictions)


def evaluate_model(model_name: str, y_true: np.ndarray, y_pred: np.ndarray, noise_name: str) -> dict:
#Calculating the experiment metrics for one model/noise conditions
    return {
        "model": model_name,
        "noise_condition": noise_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def main() -> None:
    #Running the robustness experiment
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(TEST_METADATA_PATH)
    y_test = load_pickle(Y_TEST_PATH)

    noise_conditions = {
        "original": 0.00,
        "mild_noise": 0.10,
        "moderate_noise": 0.25,
        "heavy_noise": 0.40,
    }

    results = []

    for noise_name, noise_level in noise_conditions.items():
        print(f"\nRunning condition: {noise_name}")

        noisy_texts = [
            add_text_noise(text, noise_level)
            for text in test_df["clean_text"].astype(str).tolist()
        ]

        logistic_predictions = predict_logistic(noisy_texts)
        distilbert_predictions = predict_distilbert(noisy_texts)

        results.append(
            evaluate_model(
                model_name="TF-IDF + Logistic Regression",
                y_true=y_test,
                y_pred=logistic_predictions,
                noise_name=noise_name,
            )
        )

        results.append(
            evaluate_model(
                model_name="DistilBERT",
                y_true=y_test,
                y_pred=distilbert_predictions,
                noise_name=noise_name,
            )
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(EXPERIMENT_OUTPUT_PATH, index=False)

    print("\nrobustness experiment complete")
    print(results_df)
    print(f"\nResults saved to {EXPERIMENT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()