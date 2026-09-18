"""
Email Secure Lens - Model Training Pipeline
Author: Email Secure Lens ML Team
Description:
    Trains a TF-IDF + Logistic Regression spam classification model
    on the cleaned emails dataset, evaluates its performance,
    and serializes the artifacts for production inference.
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Import reusable cleaner
try:
    from ml.preprocess import clean_text
except ImportError:
    from preprocess import clean_text


def train_spam_detector(
    data_path: str = None,
    model_output_dir: str = None,
    test_size: float = 0.20,
    random_state: int = 42,
) -> dict:
    """
    Trains and evaluates a TF-IDF + Logistic Regression model.

    Args:
        data_path (str, optional): Path to cleaned_emails.csv.
        model_output_dir (str, optional): Directory to save model & vectorizer artifacts.
        test_size (float): Proportion of dataset to include in the test split (default 0.20).
        random_state (int): Random seed for reproducible splitting and model training (default 42).

    Returns:
        dict: Evaluation metrics and training summary.
    """
    # 1. Resolve Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if data_path is None:
        data_path = os.path.join(current_dir, "cleaned_emails.csv")
    if model_output_dir is None:
        model_output_dir = current_dir

    print(f"Loading cleaned dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # Make sure text is string and labels are integer
    df["text"] = df["text"].astype(str)
    df["spam"] = df["spam"].astype(int)

    X = df["text"]
    y = df["spam"]

    # 2. Stratified Train/Test Split (Preserving class ratio)
    print(f"Splitting dataset (test_size={test_size}, random_state={random_state}, stratify=y)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    total_records = len(df)
    train_count = len(X_train)
    test_count = len(X_test)
    train_spam = int(y_train.sum())
    train_legit = int((y_train == 0).sum())
    test_spam = int(y_test.sum())
    test_legit = int((y_test == 0).sum())

    print(f"Total Records : {total_records}")
    print(f"Training set  : {train_count} records ({train_legit} legit, {train_spam} spam)")
    print(f"Testing set   : {test_count} records ({test_legit} legit, {test_spam} spam)")

    # 3. TF-IDF Feature Extraction (Fit ONLY on training data to avoid data leakage)
    print("\nFitting TF-IDF Vectorizer on training data...")
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)

    print("Transforming test data with fitted TF-IDF Vectorizer...")
    X_test_vec = vectorizer.transform(X_test)

    # 4. Train Logistic Regression Classifier
    print("Training Logistic Regression classifier...")
    model = LogisticRegression(random_state=random_state, max_iter=1000)
    model.fit(X_train_vec, y_train)

    # 5. Evaluate on Test Data
    print("Evaluating model on test dataset...")
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    clf_report = classification_report(
        y_test,
        y_pred,
        target_names=["Legitimate (0)", "Spam (1)"],
        digits=4,
    )

    # 6. Save Model and Vectorizer Artifacts
    os.makedirs(model_output_dir, exist_ok=True)
    model_path = os.path.join(model_output_dir, "model.joblib")
    vec_path = os.path.join(model_output_dir, "vectorizer.joblib")

    print(f"Saving model artifact to: {model_path}")
    joblib.dump(model, model_path)

    print(f"Saving vectorizer artifact to: {vec_path}")
    joblib.dump(vectorizer, vec_path)

    # Also save .pkl format for broad compatibility
    model_pkl_path = os.path.join(model_output_dir, "model.pkl")
    vec_pkl_path = os.path.join(model_output_dir, "vectorizer.pkl")
    joblib.dump(model, model_pkl_path)
    joblib.dump(vectorizer, vec_pkl_path)

    results = {
        "total_records": total_records,
        "train_count": train_count,
        "test_count": test_count,
        "train_legit": train_legit,
        "train_spam": train_spam,
        "test_legit": test_legit,
        "test_spam": test_spam,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
        "classification_report": clf_report,
        "model_path": model_path,
        "vectorizer_path": vec_path,
    }

    return results


def predict_single_email(
    raw_text: str,
    model_path: str = None,
    vectorizer_path: str = None,
) -> dict:
    """
    Helper function to classify a single raw email string.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if model_path is None:
        model_path = os.path.join(current_dir, "model.joblib")
    if vectorizer_path is None:
        vectorizer_path = os.path.join(current_dir, "vectorizer.joblib")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    cleaned = clean_text(raw_text)
    vec = vectorizer.transform([cleaned])
    prediction = int(model.predict(vec)[0])
    probabilities = model.predict_proba(vec)[0]

    return {
        "prediction": prediction,
        "label": "Spam" if prediction == 1 else "Legitimate",
        "spam_probability": float(probabilities[1]),
        "legitimate_probability": float(probabilities[0]),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Email Secure Lens - Model Training & Evaluation")
    print("=" * 60)

    results = train_spam_detector()

    print("\n" + "=" * 60)
    print("                EVALUATION METRICS")
    print("=" * 60)
    print(f"Accuracy  : {results['accuracy']:.4f} ({results['accuracy'] * 100:.2f}%)")
    print(f"Precision : {results['precision']:.4f} ({results['precision'] * 100:.2f}%)")
    print(f"Recall    : {results['recall']:.4f} ({results['recall'] * 100:.2f}%)")
    print(f"F1-Score  : {results['f1_score']:.4f} ({results['f1_score'] * 100:.2f}%)")
    print("\n--- Confusion Matrix ---")
    print(results["confusion_matrix"])
    print("\n--- Classification Report ---")
    print(results["classification_report"])
    print("=" * 60)
    print("Model and Vectorizer saved successfully!")
