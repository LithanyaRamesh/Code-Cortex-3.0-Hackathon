# Email Secure Lens - Machine Learning Module

This directory contains the machine learning spam classification engine for the **Email Secure Lens** application.

---

## 📌 What This Module Does

The ML module receives raw email text (subject + body), processes and cleans the text, extracts numerical features using a trained **TF-IDF Vectorizer**, and classifies the email using a trained **Logistic Regression** model into:
* `"spam"`
* `"legitimate"`

It also returns the **real probability score (confidence)** of the prediction.

---

## 📦 Requirements & Installation

Install the required Python packages:

```bash
pip install -r ml/requirements.txt
```

Or manually:
```bash
pip install scikit-learn joblib pandas
```

---

## 🚀 Quick Start for Backend Integration

### 1. Importing the Inference Function

From the project root directory, import `predict_email`:

```python
from ml.predict import predict_email

# Example 1: Check a spam email
email_text = "Congratulations! You won a prize. Click here now."
result = predict_email(email_text)
print(result)
# Output: {'prediction': 'spam', 'confidence': 0.7564}

# Example 2: Check a legitimate email
email_text = "Hi team, please find attached the weekly project status report."
result = predict_email(email_text)
print(result)
# Output: {'prediction': 'legitimate', 'confidence': 0.8951}
```

---

## 📥 Input Specification

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `email_text` | `str` | The raw text of the email (subject line, body text, or combined). |

---

## 📤 Output Specification

The function returns a standard Python dictionary:

```python
{
    "prediction": "spam" | "legitimate",  # string: 'spam' or 'legitimate'
    "confidence": 0.8112                   # float: probability between 0.0 and 1.0
}
```

* `prediction`: `"spam"` if model predicts label `1`, `"legitimate"` if label `0`.
* `confidence`: The true model confidence probability corresponding to the predicted class.

---

## 📂 File Structure

```
ml/
├── emails.csv            # Original raw email dataset
├── cleaned_emails.csv    # Cleaned dataset (duplicates removed, text normalized)
├── preprocess.py         # Text cleaning and preprocessing module (`clean_text`)
├── train.py              # Reproducible training pipeline (TF-IDF + Logistic Regression)
├── predict.py            # Inference engine with `predict_email()`
├── model.joblib          # Trained Logistic Regression classifier artifact
├── vectorizer.joblib     # Fitted TF-IDF Vectorizer artifact
├── requirements.txt      # Python dependencies
└── README.md             # This backend documentation
```

---

## ⚙️ Path Resolution

* `predict.py` automatically resolves artifact paths relative to its own location using `os.path.dirname(__file__)`.
* You can safely call `predict_email()` from **any working directory** in your backend server (e.g., FastAPI, Flask, Django).
