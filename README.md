# Concord: A Negotiation Intelligence Platform

> Modeling and simulating negotiations across competing reasoning frameworks to explore conflict, compromise, and consensus

---

## Overview

Concord is an NLP powered negotiation intelligence platform 

Traditional negotiation systems often attempt to predict a single "correct" recommendation. Concord instead analyzes the same negotiation through multiple analytical reasoning frameworks, revealing where different perspectives agree, where they disagree, and what compromises they would recommend.

Rather than replacing human judgment, Concord provides structured decision support by comparing multiple lenses of reasoning.

---

## Key Features

- Synthetic negotiation dataset generation
- NLP preprocessing and feature engineering pipeline
- Three machine learning approaches
  - Naive Baseline
  - TF-IDF + Logistic Regression
  - DistilBERT
- Multi-framework negotiation analysis
- Robustness experimentation
- Interactive Gradio web application
- Confidence visualization and downloadable reports

---

# Reasoning Frameworks

Every negotiation is analyzed using six fixed reasoning frameworks:

1. Legal & Rights Framework
2. Strategic & Economic Framework
3. Ethical Framework
4. Behavioral & Psychological Framework
5. Stakeholder & Systems Framework
6. Cultural & Social Framework

These frameworks represent analytical lenses rather than political positions or personas

------------------

# Dataset

The project uses a fully synthetic dataset generated specifically for this project.

Dataset characteristics:

- 50 unique negotiations
- 6 reasoning framework analyses per negotiation
- 300 total observations

Each row represents:

> One negotiation transcript analyzed through one reasoning framework.

Target labels:

- support
- conditional_support
- needs_more_info

Generate the dataset:

python scripts/make_dataset.py


---

# Feature Engineering

The preprocessing pipeline prepares negotiation text for downstream machine learning.

The pipeline performs:

- Text preparation
- Label encoding
- Train/test split
- TF-IDF vectorization
- Processed dataset generation

Run:

python scripts/build_features.py


Outputs:

- Processed dataset
- TF-IDF matrices
- Label encoder
- TF-IDF vectorizer

---

# Models

Concord implements all three required modeling approaches.

## 1. Naive Baseline

Majority-class classifier used as the reference baseline.

Run:

```bash
python models/baseline.py
```

---

## 2)  Classical Machine Learning

TF-IDF text representation combined with Logistic Regression.

Run:


python models/classic_logistic_regression.py


---

## 3. Deep Learning

Fine-tuned DistilBERT sequence classifier.

Run:


python models/distilbert.py


---

# Evaluation

Run:


python scripts/evaluate.py


## Final Results

| Model | Accuracy | Precision | Recall | Weighted F1 | Avg. Confidence |
|---------|----------|-----------|---------|-------------|----------------|
| Naive Baseline | 0.5000 | 0.2500 | 0.5000 | 0.3333 | — |
| TF-IDF + Logistic Regression | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 0.5629 |
| DistilBERT | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 0.8423 |

The perfect classification performance should be interpreted within the context of the synthetic dataset. To further evaluate generalization, robustness experiments were conducted under progressively noisier input conditions.

---

# Robustness Experiment

Run with-

python scripts/experiment.py


Words were progressively removed from negotiation transcripts to simulate incomplete notes and noisy real world inputs 

| Model | Original | Mild Noise | Moderate Noise | Heavy Noise |
|---------|-----------|------------|----------------|-------------|
| Logistic Regression | 100.0% | 96.67% | 95.00% | 83.33% |
| DistilBERT | 100.0% | 98.33% | 90.00% | 80.00% |

These experiments demonstrate that while both models perform perfectly on the clean synthetic dataset, performance degrades under increasingly noisy conditions, providing a more realistic assessment of model robustness.

---

# Web Application

The project includes an interactive Gradio application deployed on Render.

The application allows users to:

- Paste negotiation transcripts
- Analyze negotiations across six reasoning frameworks
- View model predictions
- Compare framework-specific recommendations
- Visualize prediction confidence
- Download analysis reports

Run locally:
python app.py


---

# Installation

Create the environment:

```bash
conda create -n concord-nlp python=3.10 -y
```

Activate:

```bash
conda activate concord-nlp
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Full Pipeline

Generate data

```bash
python scripts/make_dataset.py
```

Build features

```bash
python scripts/build_features.py
```

Train baseline

```bash
python models/baseline.py
```

Train Logistic Regression

```bash
python models/classic_logistic_regression.py
```

Train DistilBERT

```bash
python models/distilbert.py
```

Evaluate models

```bash
python scripts/evaluate.py
```

Run robustness experiment

```bash
python scripts/experiment.py
```

Launch application

```bash
python app.py
```

---

# Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Hugging Face Transformers
- DistilBERT
- Gradio
- Plotly
- PyTorch

---

# Future Work

Potential future improvements include:

- Larger and more diverse negotiation datasets
- API call layering to LLM for Concord App generated scripts
- Retrieval Augmented Generation 
- Knowledge graph integration
- Multi turn negotiation simulation
- Additional LLM assisted reasoning summaries
- Reinforcement learning for negotiation strategy optimization

---

# Notes

Generated model artifacts, processed datasets, and experiment outputs are intentionally excluded from version control to keep the repository lightweight. These artifacts can be regenerated using the provided pipeline scripts.


---

**Duke University — MEng Artificial Intelligence**

**Natural Language Processing, Dr. Brenae Bent**

**Author:** Hadil Ghazal, 2026