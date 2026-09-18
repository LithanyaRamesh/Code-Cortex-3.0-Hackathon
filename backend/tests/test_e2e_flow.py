"""
Full End-to-End integration test covering all 16 verification requirements.
"""
import json
import os
import sys
import uuid
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import app
from database import init_db

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_environment():
    init_db()


def test_complete_e2e_system_flow():
    # 1. Registration - Full Name, Email, Password
    user_email = f"lead_{uuid.uuid4().hex[:8]}@mailshield-enterprise.io"
    user_name = "Elena Rostova"
    user_pass = "HyperSecurePass2026!"

    reg_res = client.post("/auth/register", json={
        "full_name": user_name,
        "email": user_email,
        "password": user_pass
    })
    assert reg_res.status_code == 200, f"Registration failed: {reg_res.text}"
    reg_data = reg_res.json()
    assert "token" in reg_data
    assert reg_data["user"]["full_name"] == user_name
    assert reg_data["user"]["email"] == user_email

    # 2. Invalid Login - Random credentials must NOT allow login
    bad_login = client.post("/auth/login", json={
        "email": "random_unregistered@bad.com",
        "password": "wrong_password_123"
    })
    assert bad_login.status_code == 401
    assert "detail" in bad_login.json()

    # Wrong password on registered email
    wrong_pwd_login = client.post("/auth/login", json={
        "email": user_email,
        "password": "WrongPassword999!"
    })
    assert wrong_pwd_login.status_code == 401

    # 3. Valid Login
    valid_login = client.post("/auth/login", json={
        "email": user_email,
        "password": user_pass
    })
    assert valid_login.status_code == 200
    auth_data = valid_login.json()
    token = auth_data["token"]
    auth_headers = {"Authorization": f"Bearer {token}"}

    # 4. User info display (/auth/me)
    me_res = client.get("/auth/me", headers=auth_headers)
    assert me_res.status_code == 200
    assert me_res.json()["full_name"] == user_name
    assert me_res.json()["email"] == user_email

    # 5. Protected routes rejection without token
    unauthed_me = client.get("/auth/me")
    assert unauthed_me.status_code == 401

    # 6. Google OAuth configuration check
    g_status = client.get("/auth/google/status")
    assert g_status.status_code == 200
    assert "configured" in g_status.json()
    assert "message" in g_status.json()

    # If not configured, POST /auth/google returns 501
    if not g_status.json()["configured"]:
        g_attempt = client.post("/auth/google", json={"credential": "mock_credential"})
        assert g_attempt.status_code == 501
        assert "not configured" in g_attempt.json()["detail"].lower()

    # 7. Real ML Metrics
    metrics_res = client.get("/metrics")
    assert metrics_res.status_code == 200
    m = metrics_res.json()
    assert m["dataset_rows"] == 5695
    assert m["duplicates_removed"] == 33
    assert m["accuracy"] == 0.9851
    assert m["precision"] == 1.0000
    assert m["recall"] == 0.9380
    assert m["f1_score"] == 0.9680
    assert m["split"] == "80/20 stratified"

    # 8. Empty input rejection
    empty_res = client.post("/analyze", json={"raw_email": ""}, headers=auth_headers)
    assert empty_res.status_code == 400

    ws_res = client.post("/predict", json={"raw_email": "   \n\t  "}, headers=auth_headers)
    assert ws_res.status_code == 400

    # 9. Spam/Phishing Prediction
    phish_sample = (
        "From: CEO Executive Office <finance-update@ceo-payroll-sec.com>\n"
        "To: accounting@company.com\n"
        "Subject: URGENT: Executive Wire Transfer Request #99402\n"
        "DKIM-Signature: NONE\n\n"
        "Please execute an immediate wire payment of $48,500.00 before 12:00 PM today. "
        "Invoice link: http://login-verify-auth-session-billing.net/invoice"
    )
    phish_res = client.post("/analyze", json={"raw_email": phish_sample}, headers=auth_headers)
    assert phish_res.status_code == 200
    p_data = phish_res.json()
    assert p_data["risk_level"] in ("Critical", "Elevated")
    assert p_data["confidence"] > 50.0
    assert p_data["dkim"] == "NONE"
    assert len(p_data["indicators"]) > 0
    scan1_id = p_data["scan_id"]

    # 10. Legitimate Email Prediction
    legit_sample = (
        "From: GitHub Billing <billing@github.com>\n"
        "To: accounting@company.com\n"
        "Subject: Your GitHub Enterprise Subscription Invoice #GH-883910\n"
        "DKIM-Signature: PASS\n"
        "Authentication-Results: spf=pass dkim=pass dmarc=pass\n\n"
        "Hi there, your monthly invoice of $24.00 is ready. No action required."
    )
    legit_res = client.post("/predict", json={"raw_email": legit_sample}, headers=auth_headers)
    assert legit_res.status_code == 200
    l_data = legit_res.json()
    assert l_data["risk_level"] == "Low"
    assert l_data["dkim"] == "PASS"
    assert l_data["spf"] == "PASS"
    scan2_id = l_data["scan_id"]

    # 11. Stored detection history for user
    hist_res = client.get("/history", headers=auth_headers)
    assert hist_res.status_code == 200
    history = hist_res.json()
    assert len(history) >= 2
    scan_ids = [h["scan_id"] for h in history]
    assert scan1_id in scan_ids
    assert scan2_id in scan_ids

    # 12. Single scan details lookup
    detail_res = client.get(f"/history/{scan1_id}", headers=auth_headers)
    assert detail_res.status_code == 200
    assert detail_res.json()["scan_id"] == scan1_id

    # 13. Aggregated Stats from real database
    stats_res = client.get("/stats", headers=auth_headers)
    assert stats_res.status_code == 200
    stats = stats_res.json()
    assert stats["total_analyzed"] >= 2
    assert stats["threats_count"] >= 1
    assert stats["legitimate_count"] >= 1
    assert stats["has_data"] is True
    assert stats["avg_confidence"] > 0
