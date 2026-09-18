# Email Secure Lens - Code Cortex 3.0 Hackathon

**Email Secure Lens** is an intelligent email security and spam detection system developed for the **Code Cortex 3.0 Hackathon**.

---

## 📁 Repository Structure

* [`ml/`](file:///c:/Users/Asus/Downloads/secure%20email%20lens/ml/README.md) — Machine Learning pipeline, preprocessing, trained models, inference engine, and documentation.
  * `preprocess.py`: Text cleaning, lowercase conversion, and whitespace normalization pipeline.
  * `train.py`: Stratified 80/20 train/test split, TF-IDF feature extraction, and Logistic Regression training pipeline.
  * `predict.py`: Production inference module exporting `predict_email()`.
  * `model.joblib`: Serialized trained Logistic Regression spam classification model.
  * `vectorizer.joblib`: Serialized TF-IDF vectorizer.
  * `cleaned_emails.csv`: Cleaned dataset (5,695 emails).
  * `emails.csv`: Raw official spam dataset.
  * `requirements.txt`: Python ML dependencies (`scikit-learn`, `joblib`, `pandas`).
  * `README.md`: Backend integration guide.

---

## 🚀 Quick Start

### 1. Install ML Dependencies
```bash
pip install -r ml/requirements.txt
```

### 2. Run Inference
```python
from ml.predict import predict_email

result = predict_email("Congratulations! You won a prize. Click here now.")
print(result)
# Output: {'prediction': 'spam', 'confidence': 0.7564}
```

---

## 📊 Model Performance

* **Model Architecture:** TF-IDF Vectorizer + Logistic Regression (Stratified 80/20 split)
* **Dataset Size:** 5,695 cleaned emails (4,327 legitimate, 1,368 spam)
* **Accuracy:** **98.51%**
* **Precision (Spam):** **100.00%** (0 False Positives on test set)
* **Recall (Spam):** **93.80%**
* **F1-Score (Spam):** **96.80%**
