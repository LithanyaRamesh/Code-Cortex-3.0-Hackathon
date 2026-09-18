"""
Tests for the trained model artifacts and prediction sanity.
Run: pytest
"""
import json
import os
import joblib
import pytest

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")


@pytest.fixture(scope="module")
def model():
    vec = joblib.load(os.path.join(MODEL_DIR, "vectorizer.joblib"))
    clf = joblib.load(os.path.join(MODEL_DIR, "classifier.joblib"))
    return vec, clf


def test_metrics_file_exists_and_reasonable():
    with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
        m = json.load(f)
    assert m["test_rows"] > 0
    assert 0 <= m["accuracy"] <= 1
    assert 0 <= m["f1_score"] <= 1
    assert m["accuracy"] == 0.9851
    assert m["dataset_rows"] == 5695
    assert m["duplicates_removed"] == 33


def test_obvious_spam_scores_high(model):
    vec, clf = model
    text = "URGENT wire transfer $50000 immediately click http://verify-login-secure.xyz/pay"
    prob = clf.predict_proba(vec.transform([text]))[0][1]
    assert prob > 0.5


def test_obvious_ham_scores_low(model):
    vec, clf = model
    text = "Hi team, attached is the agenda for tomorrow's 10am planning meeting. Thanks!"
    prob = clf.predict_proba(vec.transform([text]))[0][1]
    assert prob < 0.5
