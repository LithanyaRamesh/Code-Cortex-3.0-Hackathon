"""
Email Secure Lens - Model Inference Module
Author: Email Secure Lens ML Team
Description:
    Loads the trained TF-IDF vectorizer and Logistic Regression model
    to perform inference on new email texts.
"""

import os
import joblib

# Import reusable cleaner
try:
    from ml.preprocess import clean_text
except ImportError:
    from preprocess import clean_text

# Global cache for loaded model and vectorizer to ensure fast inference
_MODEL = None
_VECTORIZER = None


def _load_artifacts():
    """
    Helper function to load model and vectorizer artifacts once and cache them.
    """
    global _MODEL, _VECTORIZER
    if _MODEL is None or _VECTORIZER is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Primary paths (joblib) with fallback to pkl
        model_path = os.path.join(current_dir, "model.joblib")
        if not os.path.exists(model_path):
            model_path = os.path.join(current_dir, "model.pkl")

        vec_path = os.path.join(current_dir, "vectorizer.joblib")
        if not os.path.exists(vec_path):
            vec_path = os.path.join(current_dir, "vectorizer.pkl")

        _MODEL = joblib.load(model_path)
        _VECTORIZER = joblib.load(vec_path)
    
    return _MODEL, _VECTORIZER


def predict_email(email_text: str) -> dict:
    """
    Classifies a raw email string as 'spam' or 'legitimate' and calculates
    the prediction confidence.

    Args:
        email_text (str): The raw email content/subject string.

    Returns:
        dict: A dictionary in the format:
            {
                "prediction": "spam" | "legitimate",
                "confidence": float (between 0.0 and 1.0)
            }
    """
    model, vectorizer = _load_artifacts()

    # 1. Clean email text using preprocessing function
    cleaned = clean_text(email_text)

    # 2. Transform text using fitted TF-IDF vectorizer
    transformed = vectorizer.transform([cleaned])

    # 3. Predict class (0 = legitimate, 1 = spam)
    pred_class = int(model.predict(transformed)[0])

    # 4. Predict class probabilities
    probabilities = model.predict_proba(transformed)[0]

    # 5. Extract confidence corresponding to the predicted class
    predicted_confidence = float(probabilities[pred_class])
    label = "spam" if pred_class == 1 else "legitimate"

    return {
        "prediction": label,
        "confidence": round(predicted_confidence, 4)
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Email Secure Lens - Live Inference Test")
    print("=" * 60)

    # Sample 1: Spam-like email
    spam_sample = "Congratulations! You have won a $1000 cash prize. Click here now to claim your reward."
    result_1 = predict_email(spam_sample)

    print("\n--- Test Email 1 (Spam Sample) ---")
    print(f"Email Text : \"{spam_sample}\"")
    print(f"Prediction : {result_1['prediction']}")
    print(f"Confidence : {result_1['confidence']} ({result_1['confidence'] * 100:.2f}%)")
    print(f"Output Dict: {result_1}")

    # Sample 2: Legitimate email
    legit_sample = "Hi, please find the meeting schedule attached. Let me know if you have any questions."
    result_2 = predict_email(legit_sample)

    print("\n--- Test Email 2 (Legitimate Sample) ---")
    print(f"Email Text : \"{legit_sample}\"")
    print(f"Prediction : {result_2['prediction']}")
    print(f"Confidence : {result_2['confidence']} ({result_2['confidence'] * 100:.2f}%)")
    print(f"Output Dict: {result_2}")

    print("\n" + "=" * 60)
    print("Inference tests completed successfully!")
    print("=" * 60)
