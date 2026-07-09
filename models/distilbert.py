# No AI was used to generate this code , authored by Hadil Ghazal on 7/8/26
#------------------------------------------------------
# DistilBERT deep learning model for Concord
# This script fine tunes DistilBERT for framework position classificatio and uses the processed train/test metadata created by build_features.py
#------------------------------------------------------

from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

TRAIN_METADATA_PATH = PROCESSED_DATA_DIR / "train_metadata.csv"
TEST_METADATA_PATH = PROCESSED_DATA_DIR / "test_metadata.csv"
DISTILBERT_OUTPUT_DIR = MODELS_DIR / "distilbert_concord"

MODEL_NAME = "distilbert-base-uncased"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    #Loading processed train and test metadata
    train_df = pd.read_csv(TRAIN_METADATA_PATH)
    test_df = pd.read_csv(TEST_METADATA_PATH)
    return train_df, test_df


def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    """Tokenizign teh model text for DistilBERT
    Truncation needed here because negotiation transcripts can be longer than
    the model's maximum context window
    """
    return dataset.map(
        lambda batch: tokenizer(
            batch["clean_text"],
            padding="max_length",
            truncation=True,
            max_length=256,
        ),
        batched=True,
    )


def train_distilbert() -> None:
    #Finetunimg DistilBERT for label_position classification.
    
    train_df, test_df = load_data()

    train_dataset = Dataset.from_pandas(
        train_df[["clean_text", "label_encoded"]].rename(columns={"label_encoded": "labels"})
    )
    test_dataset = Dataset.from_pandas(
        test_df[["clean_text", "label_encoded"]].rename(columns={"label_encoded": "labels"})
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = tokenize_dataset(train_dataset, tokenizer)
    test_dataset = tokenize_dataset(test_dataset, tokenizer)

    num_labels = train_df["label_encoded"].nunique()

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
    )

    training_args = TrainingArguments(
        output_dir=str(DISTILBERT_OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=str(MODELS_DIR / "distilbert_logs"),
        logging_steps=10,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    trainer.save_model(str(DISTILBERT_OUTPUT_DIR))
    tokenizer.save_pretrained(str(DISTILBERT_OUTPUT_DIR))

    print("DistilBERT model trained complete")
    print(f"saved to: {DISTILBERT_OUTPUT_DIR}")


if __name__ == "__main__":
    train_distilbert()