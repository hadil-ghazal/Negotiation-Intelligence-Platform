"""
Naive baseline model
GOal: to predict the majority class from the training set, it will provide a
minimum performance benchmark that all more sophisticated models must beat
"""

from pathlib import Path
import pickle

from sklearn.dummy import DummyClassifier


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

Y_TRAIN_PATH = PROCESSED_DATA_DIR / "y_train.pkl"
BASELINE_MODEL_PATH = MODELS_DIR / "baseline_model.pkl"


def load_pickle(file_path: Path):
    #loading pickle object from disk
    with open(file_path, "rb") as file:
        return pickle.load(file)


def save_pickle(object_to_save, file_path: Path) -> None:
    #Saving Python object as a pickle file
    with open(file_path, "wb") as file:
        pickle.dump(object_to_save, file)


def train_baseline() -> DummyClassifier:
    """ Training  a majority class baseline classifier
    where the baseline is intentionally simple for comparison
    .. answering the question: how well can we do by always predicting the most common label?
    """
    y_train = load_pickle(Y_TRAIN_PATH)

    baseline_model = DummyClassifier(strategy="most_frequent")
    baseline_model.fit([[0]] * len(y_train), y_train)

    save_pickle(baseline_model, BASELINE_MODEL_PATH)

    print("Baseline model trained")
    print(f" saved to: {BASELINE_MODEL_PATH}")

    return baseline_model


if __name__ == "__main__":
    train_baseline()