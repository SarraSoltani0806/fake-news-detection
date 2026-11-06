# Fake News Detection with Machine Learning

A Natural Language Processing (NLP) classification project that predicts whether a news article belongs to the **REAL** or **FAKE** class using **TF-IDF** features and one linear classifier trained with the **Passive-Aggressive** learning algorithm.

> **Important:** The model learns patterns in labeled articles; it does **not** independently fact-check claims or establish whether a news story is true.

## Project overview

This project follows an end-to-end supervised machine learning workflow:

1. Explore and inspect the news dataset with Pandas.
2. Check missing data, label distribution, text lengths, and duplicates.
3. Prepare the `text` feature and `label` target.
4. Split the data into training (80%) and testing (20%) sets, stratified by label.
5. Learn TF-IDF features **from the training set only** to avoid leakage.
6. Train a single Passive-Aggressive linear classifier.
7. Evaluate on held-out data using accuracy, precision, recall, F1-score, and a confusion matrix.
8. Inspect classification errors and influential TF-IDF features.

## Dataset

The working dataset contains **20,000 articles** after cleaning, with two modeling columns:

| Column | Description |
| --- | --- |
| `text` | News article content (model input) |
| `label` | `REAL` or `FAKE` (prediction target) |

The initial dataset also includes `title`, `date`, `source`, `author`, and `category`; the modeling baseline uses **only article text**. During exploration, `source` and `author` each had 1,000 missing values, but these fields were not required for this baseline. Checks found no exact duplicate texts or identical texts with conflicting labels.

**Data provenance:** Add the original dataset URL, creator, and license here before publishing or redistributing the dataset. If its redistribution terms are unclear, keep the CSV out of Git and document how to obtain it.

## Approach

```text
Labeled news articles
        |
        v
Data inspection & cleaning
        |
        v
Stratified train/test split (80/20)
        |
        v
TF-IDF vectorization
(fit on training data; transform test data)
        |
        v
Passive-Aggressive classifier
        |
        v
Predictions: REAL / FAKE
        |
        v
Evaluation and error analysis
```

### TF-IDF

`TfidfVectorizer` represents each article as a sparse numerical feature vector. It upweights terms that distinguish a document from others in the training corpus. The baseline uses English stop-word filtering and `max_df=0.7` (excludes terms appearing in more than 70% of training documents).

### Classifier

One classifier is used throughout: a **Passive-Aggressive linear classifier**. It adjusts the decision boundary when examples violate the classification margin and otherwise leaves the weights unchanged.

**Compatibility note:** `PassiveAggressiveClassifier` is deprecated in scikit-learn 1.8. For newer versions, the documented PA-I equivalent uses:

```python
from sklearn.linear_model import SGDClassifier

model = SGDClassifier(
    loss="hinge",
    penalty=None,
    learning_rate="pa1",
    eta0=1.0,
    max_iter=1000,
    random_state=42,
)
```

This keeps the **same Passive-Aggressive learning approach**; it does not add a second model.

## Project structure

```text
fake-news-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── fake_news_detection.ipynb   # Add your Jupyter notebook
├── data/
│   └── fake_news_dataset.csv   # Local file, ignored by Git
└── models/                      # Generated model files, ignored by Git
```

## Setup and execution

Prerequisites: Python 3.10+ and JupyterLab or VS Code with the Jupyter extension.

```bash
# From the repository directory
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Open `fake_news_detection.ipynb` and run all cells in order. Update the CSV path in the notebook to match where you saved your dataset, for example:

```python
import pandas as pd

df = pd.read_csv("data/fake_news_dataset.csv")
```

## Evaluation

The training run shown in development used **763 TF-IDF features** and completed in **10 passes** over the training data. These figures describe training, **not prediction quality**.

Fill in the following table after evaluating on the held-out test set; results have not yet been provided:

| Metric | Test result |
| --- | --- |
| Accuracy | _To be measured_ |
| Precision (FAKE) | _To be measured_ |
| Recall (FAKE) | _To be measured_ |
| F1-score (FAKE) | _To be measured_ |

Also include your confusion matrix plot and a short analysis of false positives and false negatives.

## Limitations and next steps

- Predictions reflect patterns in the training labels, not independent verification of factual claims.
- Text models may learn artifacts of publisher, topic, writing style, or collection period.
- Exact deduplication does not remove near-duplicates; grouped or time-based evaluation can provide a more demanding generalization test.
- Future work **with the same classifier** could include hyperparameter tuning, n-gram features, cross-validation, and a small user interface.

## Technology

Python · Pandas · NumPy · Matplotlib · Seaborn · scikit-learn · JupyterLab

## Status

**In progress.** Data exploration, cleaning, TF-IDF transformation, and classifier training have been carried out. Final test metrics, plots, and any interface will be added after evaluation.
