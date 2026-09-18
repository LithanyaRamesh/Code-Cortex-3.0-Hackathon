import io
import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import app
from database import init_db

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup():
    init_db()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_metrics():
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    assert "accuracy" in body
    assert "precision" in body
    assert "recall" in body
    assert "f1_score" in body
    assert body["accuracy"] == 0.9851
    assert body["dataset_rows"] == 5695
    assert body["duplicates_removed"] == 33


def test_analyze_empty_rejected():
    r = client.post("/analyze", json={"raw_email": ""})
    assert r.status_code == 400


def test_predict_empty_rejected():
    r = client.post("/predict", json={"raw_email": "   "})
    assert r.status_code == 400


def test_analyze_phishing_sample():
    sample = (
        "From: CEO Executive Office <finance-update@ceo-payroll-sec.com>\n"
        "DKIM-Signature: NONE\n"
        "Subject: URGENT: Executive Wire Transfer Request\n\n"
        "Please wire $48,500 immediately to http://login-verify-auth-session-billing.net/pay"
    )
    r = client.post("/analyze", json={"raw_email": sample})
    assert r.status_code == 200
    body = r.json()
    assert body["risk_level"] in ("Critical", "Elevated")
    assert body["dkim"] == "NONE"
    assert len(body["indicators"]) > 0
    assert "scan_id" in body
    assert "explanation" in body


def test_predict_structured_subject_and_body():
    r = client.post("/predict", json={
        "subject": "URGENT: Immediate invoice payment required",
        "body": "Please process the wire transfer of $25,000 before noon today to our vendor."
    })
    assert r.status_code == 200
    body = r.json()
    assert "scan_id" in body
    assert "extracted_features" in body
    assert len(body["indicators"]) > 0


def test_predict_legit_sample():
    sample = (
        "From: GitHub Billing <billing@github.com>\n"
        "DKIM-Signature: PASS\n"
        "Subject: Your invoice is ready\n\n"
        "Hi, your monthly invoice for $24.00 is attached. No action needed."
    )
    r = client.post("/predict", json={"raw_email": sample})
    assert r.status_code == 200
    body = r.json()
    assert body["dkim"] == "PASS"
    assert body["risk_level"] == "Low"
    assert "LIKELY LEGITIMATE" in body["verdict"]


def test_predict_legitimate_normal_emails():
    # 1. College assignment reminder
    r1 = client.post("/predict", json={
        "subject": "Assignment 3 submission deadline reminder",
        "body": "Dear students, this is a reminder that Assignment 3 is due this Friday at 11:59 PM on Canvas. Office hours are open."
    })
    assert r1.status_code == 200
    assert r1.json()["risk_level"] == "Low"
    assert "LIKELY LEGITIMATE" in r1.json()["verdict"]

    # 2. Team meeting
    r2 = client.post("/predict", json={
        "subject": "Team sync tomorrow at 10 AM",
        "body": "Hi team, let us meet in Conference Room B tomorrow to go over the sprint backlog. Please bring your project updates."
    })
    assert r2.status_code == 200
    assert r2.json()["risk_level"] == "Low"
    assert "LIKELY LEGITIMATE" in r2.json()["verdict"]

    # 3. Amazon shipping notification
    r3 = client.post("/predict", json={
        "subject": "Your Amazon order #402-991823 has shipped",
        "body": "Your package containing Computer Monitor has shipped and will arrive on Thursday. Track your package at https://www.amazon.com/orders"
    })
    assert r3.status_code == 200
    assert r3.json()["risk_level"] == "Low"
    assert "LIKELY LEGITIMATE" in r3.json()["verdict"]

    # 4. Personal communication
    r4 = client.post("/predict", json={
        "subject": "Dinner plans this weekend",
        "body": "Hey Alex, are we still meeting for dinner on Saturday at 7 PM? Let me know which restaurant you prefer."
    })
    assert r4.status_code == 200
    assert r4.json()["risk_level"] == "Low"
    assert "LIKELY LEGITIMATE" in r4.json()["verdict"]


def test_file_upload_txt_safe():
    txt_content = (
        "From: IT Support <it@internal-helpdesk.com>\n"
        "Subject: System Maintenance Notice\n\n"
        "Scheduled server maintenance will occur this Saturday from 2am to 4am."
    )
    file_bytes = io.BytesIO(txt_content.encode("utf-8"))
    r = client.post("/predict/file", files={"file": ("maintenance.txt", file_bytes, "text/plain")})
    assert r.status_code == 200
    assert "scan_id" in r.json()
    assert r.json()["risk_level"] == "Low"


def test_file_upload_eml_phishing():
    eml_content = (
        "From: Accounting <payroll@spoofed-sec-portal.xyz>\n"
        "To: employee@corp.com\n"
        "Subject: ACTION REQUIRED: Update direct deposit account\n"
        "DKIM-Signature: fail\n"
        "Authentication-Results: dkim=fail spf=fail\n"
        "Content-Type: text/plain\n\n"
        "Click here immediately to confirm your banking details: http://verify-payroll-login.xyz/auth"
    )
    file_bytes = io.BytesIO(eml_content.encode("utf-8"))
    r = client.post("/analyze/file", files={"file": ("payroll_alert.eml", file_bytes, "message/rfc822")})
    assert r.status_code == 200
    body = r.json()
    assert body["risk_level"] in ("Critical", "Elevated")
    assert body["dkim"] == "FAIL"


def test_file_upload_unsupported_type_rejected():
    file_bytes = io.BytesIO(b"%PDF-1.4 simulated pdf")
    r = client.post("/predict/file", files={"file": ("document.pdf", file_bytes, "application/pdf")})
    assert r.status_code == 400
    assert "unsupported file type" in r.json()["detail"].lower()


def test_history_and_stats():
    # History endpoint
    hist_resp = client.get("/history")
    assert hist_resp.status_code == 200
    history = hist_resp.json()
    assert isinstance(history, list)

    # Stats endpoint
    stats_resp = client.get("/stats")
    assert stats_resp.status_code == 200
    stats = stats_resp.json()
    assert "total_analyzed" in stats
    assert "threats_count" in stats
    assert "legitimate_count" in stats
    assert "avg_confidence" in stats
    assert "daily_volume" in stats


def test_feedback_and_learning_cycle():
    # 1. Analyze an email
    scan = client.post("/predict", json={
        "subject": "System access verification notice",
        "body": "Please confirm your security preferences for quarterly update."
    }).json()
    scan_id = scan["scan_id"]
    assert "evidence_items" in scan

    # 2. Submit feedback "Actually Safe"
    fb_resp = client.post("/feedback", json={
        "scan_id": scan_id,
        "actual_label": "SAFE",
        "original_verdict": scan["verdict"],
        "subject_snippet": "System access verification notice",
        "comments": "Internal routine check"
    })
    assert fb_resp.status_code == 200
    assert fb_resp.json()["success"] is True

    # 3. Check feedback stats
    stats_resp = client.get("/feedback/stats")
    assert stats_resp.status_code == 200
    assert stats_resp.json()["total_feedback"] >= 1


def test_intelligence_and_alerts():
    # Intelligence endpoint
    intel_resp = client.get("/intelligence")
    assert intel_resp.status_code == 200
    intel = intel_resp.json()
    assert "security_score" in intel
    assert "feedback_stats" in intel
    assert "attack_vectors" in intel

    # Alerts endpoint
    alerts_resp = client.get("/alerts")
    assert alerts_resp.status_code == 200
    alerts = alerts_resp.json()
    assert isinstance(alerts, list)


def test_delete_scan_and_purge():
    # Create a scan
    scan = client.post("/analyze", json={"subject": "Test to delete", "body": "Clean text for deletion test."}).json()
    scan_id = scan["scan_id"]

    # Delete single scan
    del_resp = client.delete(f"/history/{scan_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["success"] is True

    # Verify 404 on re-fetch
    fetch_resp = client.get(f"/history/{scan_id}")
    assert fetch_resp.status_code == 404

