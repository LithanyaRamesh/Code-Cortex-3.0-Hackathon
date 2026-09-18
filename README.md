# Email Secure Lens / MailShield AI

Enterprise email spam & phishing threat defense platform built for the **Code Cortex 3.0 Hackathon**, featuring a high-fidelity cybersecurity UI, backed by a **production-grade trained Machine Learning classifier**, **SQLite persistence**, and **secure authentication**.

---

## 📁 Repository Structure

```
.
├── frontend/
│   └── index.html          # Dark cybersecurity UI (Dashboard, Analyzer, Reports, Settings, Auth)
├── backend/
│   ├── app.py               # FastAPI service with auth, ML inference, and telemetry endpoints
│   ├── auth.py              # PBKDF2 hashing, JWT generation/verification, Google OAuth helpers
│   ├── database.py          # SQLite database schema, CRUD, and telemetry aggregators
│   ├── train_model.py       # Model training with deduplication & metrics export
│   ├── requirements.txt
│   ├── data/
│   │   └── emails.csv       # Corpus dataset
│   ├── model/
│   │   ├── vectorizer.joblib
│   │   ├── classifier.joblib
│   │   └── metrics.json     # Test benchmark metrics
│   └── tests/
│       ├── test_api.py
│       ├── test_auth.py
│       ├── test_e2e_flow.py
│       └── test_model.py
├── ml/
│   ├── README.md            # Backend developer integration guide
│   ├── __init__.py          # Package initializer
│   ├── requirements.txt     # Python ML dependencies
│   ├── preprocess.py        # Text cleaning pipeline (clean_text)
│   ├── train.py             # Reproducible ML training pipeline
│   ├── predict.py           # Production inference module (predict_email)
│   ├── model.joblib         # Trained Logistic Regression model artifact
│   ├── model.pkl            # Model artifact
│   ├── vectorizer.joblib    # Trained TF-IDF Vectorizer artifact
│   ├── vectorizer.pkl       # Vectorizer artifact
│   ├── emails.csv           # Original dataset
│   └── cleaned_emails.csv   # Cleaned dataset (5,695 records)
├── docs/
│   └── architecture.md      # System architecture documentation
├── .gitignore
└── README.md
```

---

## 📊 ML Model Performance

* **Model Architecture:** TF-IDF Vectorizer + Logistic Regression (Stratified 80/20 split)
* **Dataset Size:** 5,695 cleaned emails (4,327 legitimate, 1,368 spam)
* **Accuracy:** **98.51%**
* **Precision (Spam):** **100.00%** (0 False Positives on test set)
* **Recall (Spam):** **93.80%**
* **F1-Score (Spam):** **96.80%**

---

## 🚀 Getting Started

### 1. ML Module Quick Start
```bash
pip install -r ml/requirements.txt
```
```python
from ml.predict import predict_email

result = predict_email("Congratulations! You won a prize. Click here now.")
print(result)
# Output: {'prediction': 'spam', 'confidence': 0.7564}
```

### 2. Backend Setup
```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

### 3. Frontend Launch
Open `frontend/index.html` in your web browser.

---

## 🧪 Running Automated Tests

```bash
cd backend
python -m pytest
```
