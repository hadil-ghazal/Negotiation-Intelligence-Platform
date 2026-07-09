
"""
# No AI was used to generate this code , authored by Hadil Ghazal on 7/8/26
This code script is to prepares the synthetic negotiation dataset for downstream NLP modeling and it creates cleaned text, encodes the prediction label, creates a shared train/test
split, and saves TF-IDF features for the classical ML model.

"""

from pathlib import Path
import re
import pickle
 
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
 
 
#---------------------------------------------------------------------
# Project paths
#------------------------------------------------------------------
 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
 
DATA_DIR = PROJECT_ROOT /"data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = DATA_DIR /"outputs"
MODELS_DIR = PROJECT_ROOT / "models"
 
RAW_DATA_PATH = RAW_DATA_DIR / "concord_negotiation_dataset.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "concord_processed_dataset.csv"
 
TFIDF_TRAIN_PATH = PROCESSED_DATA_DIR / "X_train_tfidf.npz"
TFIDF_TEST_PATH = PROCESSED_DATA_DIR / "X_test_tfidf.npz"
 
Y_TRAIN_PATH = PROCESSED_DATA_DIR / "y_train.pkl"
Y_TEST_PATH = PROCESSED_DATA_DIR / "y_test.pkl"
 
TRAIN_METADATA_PATH = PROCESSED_DATA_DIR / "train_metadata.csv"
TEST_METADATA_PATH = PROCESSED_DATA_DIR / "test_metadata.csv"
 
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
 
 
 
#---------------------------------------------------------------------
#-----dataset columns ------------
# ---------------------------------------------------------------------
 
REQUIRED_COLUMNS = [
    "negotiation_id",
    "transcript",
    "framework",
    "framework_interpretation",
    "label_position",
    "recommended_negotiation_move",
    "compromise_path",
]
# NOTE: framework_interpretation, recommended_negotiation_move, and
# compromise_path are still required/validated and still saved to the
# processed CSV and train/test metadata files below, since they're useful
# for manual inspection and the report's Error Analysis section. They are
# deliberately NOT included in build_model_text() — see that function's
# docstring for why (label leakage / train-inference mismatch).
 
 
#---------------------------------------------------------------------
# Helper functions
#---------------------------------------------------------------------
 
def ensure_directories() -> None:
    # creating output directories in case they don't already exist, already created the raw data using make_dataset
    # but the feaure pipeline needs processed data too and the model artifact folders
 
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
 
 
def load_dataset(file_path: Path) -> pd.DataFrame:
#loading raw dataset, raising error if not found
    if not file_path.exists():
        raise FileNotFoundError(
            f"dataset not found at {file_path}\n"
        )
 
    return pd.read_csv(file_path)
 
 
def validate_columns(df: pd.DataFrame) -> None:
#Validating that the dataset contains the minimum columns needed for modeling.
#failing early here because silent column errors can create confusing model failures later in the pipeline
    
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
 
    if missing_columns:
        raise ValueError(
            "The dataset is missing required columns:\n"
            f"{missing_columns}\n\n"
            f"the available columns are:\n{list(df.columns)}"
        )
 
 
def clean_text(text: str) -> str:
    
   # Applying conservative NLP text cleaning , intentionally avoiding stemming, lemmatization, and stopword removal because
   #... negotiation meaning often depends on words like "not", "must", "should", and "may". DistilBERT will benefit from minimally altered language
    
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s.,;:!?'-]", "", text)
    return text.strip()
 
 
def build_model_text(df: pd.DataFrame) -> pd.Series:
    """Combine the negotiation transcript and framework name into one model input.
 
    IMPORTANT: this must only use information that is genuinely available at
    inference time in app.py — the framework and the transcript. An earlier
    version of this function also folded in `framework_interpretation`,
    `recommended_negotiation_move`, and `compromise_path`. Those columns hold
    real, per-example values in this CSV that are strongly correlated with
    (and likely near-proxies for) `label_position` — training on them is
    label leakage, which is almost certainly why the classical and DistilBERT
    models evaluated at ~100% accuracy. At inference time, app.py has no way
    to know a transcript's "real" recommended move or compromise path before
    predicting one — those fields don't exist yet for a brand-new negotiation.
    Training on fields the deployed app can never supply teaches the model to
    lean on information it will never have in production, which is worse
    than merely optimistic metrics: it makes the deployed model largely
    insensitive to the actual transcript text (confirmed empirically — see
    the app.py fix from earlier this session). Removing them forces the
    model to learn from what's actually available: the framework and the
    transcript text, keeping this function identical in shape to app.py's
    build_model_text() so there's no train/inference mismatch.
    """
 
    return (
        "Framework: "
        + df["framework"].astype(str)
        + "\n\nNegotiation Transcript: "
        + df["transcript"].astype(str)
    )
 
 
def save_pickle(object_to_save, file_path: Path) -> None:
 
#Save python objectas a pickle file (for sklearn encoders and the vectorizers reused udring training)
 
    with open(file_path, "wb") as file:
        pickle.dump(object_to_save, file)
 
 
def main() -> None:
    
    #Runnign the full feature engineering pipeline
   
    ensure_directories()
 
    print(f"Loading dataset from: {RAW_DATA_PATH}")
    df = load_dataset(RAW_DATA_PATH)
    validate_columns(df)
 
    print(f"Loaded rows: {len(df)}")
    print(f"Unique negotiations: {df['negotiation_id'].nunique()}")
   
    print(f"Unique frameworks: {df['framework'].nunique()}")
    # Building the main NLP input
    #.. keeping both original and cleaned text so transformer models can later use the less processed language while classical models use cleaned text
    df["model_text"] = build_model_text(df)
    df["clean_text"] = df["model_text"].apply(clean_text)
 
    # Encoding target label to predict whether a framework supports, conditionally supports, or needs more info for a negotiation position
    label_encoder = LabelEncoder()
    df["label_encoded"] = label_encoder.fit_transform(df["label_position"])
 
    print("\nEncoded labels:")
    for label_name, label_id in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):
        print(f"  {label_id}: {label_name}")
 
    # Stratified split to keep label proportions stable across train and test sets
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["label_encoded"],
    )
 
    print(f"\nTrain rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
 
 
    #-----------------------------------------------------------------------------
    # TF-IDF is the representation for the classical ML model
    # Bigrams being included because negotiation signals are often appearing in phrases like
    # ... "needs more", "legal obligation", "mutual benefit", or "power imbalance"
    #-----------------------------------------------------------------------------
 
    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )
 
    X_train_tfidf = tfidf_vectorizer.fit_transform(train_df["clean_text"])
    X_test_tfidf = tfidf_vectorizer.transform(test_df["clean_text"])
 
    y_train = train_df["label_encoded"].to_numpy()
    y_test = test_df["label_encoded"].to_numpy()
 
    # Saving processed full dataset for inspection and later app use
    df.to_csv(PROCESSED_DATA_PATH, index=False)
 
    # Saving sparse TF-IDF matrices efficiently.
    sparse.save_npz(TFIDF_TRAIN_PATH, X_train_tfidf)
    sparse.save_npz(TFIDF_TEST_PATH, X_test_tfidf)
 
    # Saving labels
    save_pickle(y_train, Y_TRAIN_PATH)
    save_pickle(y_test, Y_TEST_PATH)
 
    # Saving the metadata so later evaluation can inspect errors by negotiation framework, domain, and other resaerch relevnt fields.
    train_df.to_csv(TRAIN_METADATA_PATH, index=False)
    test_df.to_csv(TEST_METADATA_PATH, index=False)
 
    # Saving reusable sklearn artifacts
    save_pickle(label_encoder, LABEL_ENCODER_PATH)
    save_pickle(tfidf_vectorizer, TFIDF_VECTORIZER_PATH)
 
    print("\nFeature engineering complete.")
    print(f"Processed dataset saved to: {PROCESSED_DATA_PATH}")
    print(f"TF-IDF train matrix saved to: {TFIDF_TRAIN_PATH}")
    print(f"TF-IDF test matrix saved to: {TFIDF_TEST_PATH}")
    print(f"Label encoder saved to: {LABEL_ENCODER_PATH}")
    print(f"TF-IDF vectorizer saved to: {TFIDF_VECTORIZER_PATH}")
    print(f"\nTF-IDF train shape: {X_train_tfidf.shape}")
    print(f"TF-IDF test shape: {X_test_tfidf.shape}")
 
 
if __name__ == "__main__":
    main()