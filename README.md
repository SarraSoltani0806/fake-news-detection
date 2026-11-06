# Fake News Detection — Data Science Learning Project

A hands-on **Data Science and Natural Language Processing (NLP)** project exploring text feature extraction, supervised classification, and model evaluation. The goal is to classify articles as `FAKE` or `REAL` using **TF-IDF** features and a single **Passive–Aggressive learning algorithm**.

This is a learning exercise, not a fact-checking service: the classifier learns patterns in the training data and does not verify the truth of an article.

## Dataset

The cleaned dataset contains **20,000 articles** and two modeling columns:

- `text`: article content (input)
- `label`: `FAKE` or `REAL` (target)

Exploratory analysis covered class distribution, article lengths, missing values and duplicates. The original `source` and `author` columns contained missing values and were not used as features. No conflicting labels were found for identical texts in the cleaned dataset.

## Methodology

1. **Exploratory data analysis:** inspect data structure, class distribution, missing values, duplicates and text lengths.
2. **Data preprocessing:** select article text and labels; remove invalid or duplicate observations.
3. **Train/test split:** stratified 80/20 split (16,000 / 4,000 articles).
4. **Feature extraction:** use `TfidfVectorizer` to convert articles into numerical vectors (**763 features** in this experiment). Fit the vectorizer only on the training data to avoid test leakage.
5. **Classification:** train one linear classifier using the **Passive–Aggressive (PA-I)** update rule, implemented with `SGDClassifier(loss='hinge', penalty=None, learning_rate='pa1', eta0=1.0)`.
6. **Evaluation:** analyze accuracy, per-class precision/recall/F1, the confusion matrix, misclassified examples and influential vocabulary features.

## Results

Evaluation on **4,000 held-out articles**:

| Metric | Value |
| --- | ---: |
| Accuracy | **50.42%** |
| Macro F1-score | **0.48** |
| Correct predictions | **2,017** |
| Incorrect predictions | **1,983** |

### Classification report

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| FAKE | 0.50 | 0.71 | 0.59 | 2,011 |
| REAL | 0.50 | 0.29 | 0.37 | 1,989 |

### Confusion matrix

Rows are actual classes; columns are predicted classes.

| Actual / Predicted | FAKE | REAL |
| --- | ---: | ---: |
| **FAKE** | 1,436 | 575 |
| **REAL** | 1,408 | 581 |

**Interpretation:** The model identifies many FAKE-labeled articles, but frequently misclassifies REAL-labeled articles as FAKE. Accuracy is only slightly above 50% on this nearly balanced test set, so the current classifier is **a baseline, not a reliable detection system**.

The vocabulary has only 763 features, which is worth investigating along with text quality, dataset characteristics, and the preprocessing pipeline. No improved performance is claimed here.

## Technologies

Python · Jupyter Notebook · pandas · NumPy · Matplotlib · Seaborn · scikit-learn · joblib

## Run locally

1. Clone this repository and open a terminal in its root directory.
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Place the dataset in `Data/` and adjust the CSV path in the notebook if needed.
5. Start Jupyter and run `fakenewsdetection.ipynb` from top to bottom:
   ```bash
   jupyter lab
   ```

The dataset is not included in the repository by default; ensure that you have permission to redistribute it before publishing it.

## Project structure

```text
.
├── Data/                         # local dataset (not committed)
├── models/                       # locally saved model artifacts
├── fakenewsdetection.ipynb       # analysis, training and evaluation
├── README.md
└── requirements.txt
```


