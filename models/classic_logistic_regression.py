"""Classical ML model using TF-IDF features with Logistic Regression , acting as strong and
interpretable baseline for text classification, to compare against baseline and optimized model
"""

from pathlib import Path
import pickle

from scipy import sparse
from sklearn.linear_model import LogisticRegression


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

X_TRAIN_PATH = PROCESSED_DATA_DIR / "X_train_tfidf.npz"
Y_TRAIN_PATH = PROCESSED_DATA_DIR / "y_train.pkl"
LOGISTIC_MODEL_PATH = MODELS_DIR / "logistic_regression_model.pkl"


def load_pickle(file_path: Path):
    with open(file_path, "rb") as file:
        return pickle.load(file)


def save_pickle(object_to_save, file_path: Path) -> None:
    #saving Python object as pickle file
    with open(file_path, "wb") as file:
        pickle.dump(object_to_save, file)


def train_logistic_regression() -> LogisticRegression:
    #Training a TF-IDF and Logistic Regression classifier
    # Logistic Regression is used because it is lightweight, explainable, and
    # effective for sparse text features
        
    X_train = sparse.load_npz(X_TRAIN_PATH)
    y_train = load_pickle(Y_TRAIN_PATH)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(X_train, y_train)

    save_pickle(model, LOGISTIC_MODEL_PATH)

    print("logistic Regression model trained.")
    print(f"Saved to: {LOGISTIC_MODEL_PATH}")

    return model


if __name__ == "__main__":
    train_logistic_regression()