# Financial Sentiment Analysis

## Overview

This project classifies financial text into three sentiment categories: **Positive, Negative, and Neutral**.

It compares a traditional Machine Learning approach using **TF-IDF + Logistic Regression** with **FinBERT**, a pre-trained language model designed for financial text.

## Dataset

The dataset contains **5,842 financial sentences** with the following sentiment distribution:

* Neutral: 3,130
* Positive: 1,852
* Negative: 860

Columns:

* `Sentence` — Financial text
* `Sentiment` — Sentiment label

## Approach

### 1. Data Preprocessing

* Loaded the financial sentiment dataset using Pandas
* Checked dataset structure and missing values
* Analyzed sentiment distribution
* Split the data into **80% training and 20% testing**

### 2. Traditional Machine Learning

* Converted text into numerical features using **TF-IDF**
* Trained a **Logistic Regression** classifier
* Trained a second Logistic Regression model using `class_weight="balanced"` to address class imbalance

### 3. FinBERT

Used **ProsusAI/FinBERT**, a pre-trained financial language model, to classify the same test dataset.

## Results

| Model                        |   Accuracy |  Macro F1 |
| ---------------------------- | ---------: | --------: |
| Logistic Regression          |     69.97% |     ~0.56 |
| Balanced Logistic Regression |    ~69.03% |     ~0.64 |
| **FinBERT**                  | **75.11%** | **~0.73** |

FinBERT achieved the best overall performance, demonstrating the benefit of using a domain-specific language model for financial sentiment classification.

## Confusion Matrix

The FinBERT confusion matrix is included in:

`confusion_matrix.png`

## Project Structure

```text
Financial-Sentiment-Analysis/
│
├── data.csv
├── Financial_Sentiment_Analysis.ipynb
├── inference.py
├── sentiment_model.pkl
├── tfidf_vectorizer.pkl
├── metrics.csv
├── confusion_matrix.png
└── README.md
```

## Inference

The saved Logistic Regression model can be used to classify new financial sentences through:

```bash
python inference.py
```

The inference script loads the saved model and TF-IDF vectorizer and returns the predicted sentiment.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* PyTorch
* Hugging Face Transformers
* FinBERT
* Jupyter Notebook

## Key Takeaway

The project shows that while traditional TF-IDF-based Machine Learning provides a strong baseline, **FinBERT performs better on financial sentiment classification** because it is specifically trained for financial language understanding.

