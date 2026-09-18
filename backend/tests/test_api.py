import io
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


def test_predict_structured_with_sender_option():
    r = client.post("/predict", json={
        "sender": "payroll-update@security-alert-verify.com",
        "subject": "ACTION REQUIRED: Confirm credentials",
        "body": "Please verify your account password immediately at http://login-verify-auth-session-billing.net/login."
    })
    assert r.status_code == 200
    body = r.json()
    assert "scan_id" in body
    assert body["risk_level"] in ("Critical", "Elevated")
    assert body["attacker_intent"] is not None
    assert len(body["attack_surface_vectors"]) == 8


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


def test_attack_surface_detection():
    # Threat email with links, OTP, password, and wire transfer requests
    threat_email = (
        "Subject: URGENT: Complete wire payment and verify account password\n\n"
        "Please enter your password and enter the 6-digit OTP code immediately to authorize "
        "wire payment of $15,000 to http://secure-update-portal.xyz/login. See attached invoice.pdf."
    )
    resp = client.post("/analyze", json={"raw_email": threat_email})
    assert resp.status_code == 200
    data = resp.json()

    assert "attack_surface_vectors" in data
    assert len(data["attack_surface_vectors"]) == 8
    
    # Check that link, password, otp, payment, and attachment are detected
    detected_keys = [v["key"] for v in data["attack_surface_vectors"] if v["detected"]]
    assert "link" in detected_keys
    assert "password" in detected_keys
    assert "otp" in detected_keys
    assert "payment" in detected_keys
    assert "attachment" in detected_keys

    # Check What Can Happen and What Should I Do Now
    assert "what_can_happen" in data
    assert len(data["what_can_happen"]) > 20
    assert "what_to_do_now" in data
    assert len(data["what_to_do_now"]) >= 4

    # Check Interaction Risk Scores
    assert "interaction_risk_scores" in data
    assert data["interaction_risk_scores"]["made_payment"] == 100.0
    assert data["interaction_risk_scores"]["entered_otp"] == 98.0

    # Check Attacker Intent
    assert "attacker_intent" in data
    assert data["attacker_intent"] is not None
    assert "Steal OTP" in data["attacker_intent"]["intent_title"] or "Steal Password" in data["attacker_intent"]["intent_title"]
    assert len(data["attacker_intent"]["evidence"]) >= 1
    assert len(data["attacker_intent"]["description"]) > 10


def test_attacker_intent_detection():
    # 1. OTP harvesting
    r_otp = client.post("/predict", json={
        "subject": "Security Check",
        "body": "Please authenticate by entering the 6-digit OTP code sent to your phone."
    })
    assert r_otp.status_code == 200
    d_otp = r_otp.json()
    assert d_otp["attacker_intent"]["intent_title"] == "Steal OTP / 2FA Token"
    assert d_otp["attacker_intent"]["primary_vector"] == "🔢 OTP"

    # 2. Password reset lure
    r_pwd = client.post("/predict", json={
        "subject": "Password reset notice",
        "body": "Please enter your password to confirm your security update at http://phishing-portal.xyz/login"
    })
    assert r_pwd.status_code == 200
    d_pwd = r_pwd.json()
    assert "Steal Password" in d_pwd["attacker_intent"]["intent_title"]
    assert d_pwd["attacker_intent"]["primary_vector"] == "🔑 Password"

    # 3. Financial wire transfer
    r_wire = client.post("/predict", json={
        "subject": "Executive Wire Request",
        "body": "Please process a wire transfer payment of $45,000 using the following wiring instructions."
    })
    assert r_wire.status_code == 200
    d_wire = r_wire.json()
    assert "Get Payment" in d_wire["attacker_intent"]["intent_title"]
    assert d_wire["attacker_intent"]["primary_vector"] == "💳 Payment"

    # 4. Attachment lure
    r_att = client.post("/predict", json={
        "subject": "Invoice Enclosed",
        "body": "Please open attached invoice.zip to review pending purchase order details."
    })
    assert r_att.status_code == 200
    d_att = r_att.json()
    assert "Attachment" in d_att["attacker_intent"]["intent_title"]
    assert d_att["attacker_intent"]["primary_vector"] == "📎 Attachment"

    # 5. Sensitive data lure
    r_sens = client.post("/predict", json={
        "subject": "Tax Form Update",
        "body": "Kindly provide your SSN and social security tax form w-2 for payroll verification."
    })
    assert r_sens.status_code == 200
    d_sens = r_sens.json()
    assert "Sensitive Information" in d_sens["attacker_intent"]["intent_title"]
    assert d_sens["attacker_intent"]["primary_vector"] == "📤 Sensitive Information"

    # 6. Benign email
    r_clean = client.post("/predict", json={
        "subject": "Team Lunch",
        "body": "Hey team, let's meet at 12:30 PM for lunch in the 3rd floor cafeteria."
    })
    assert r_clean.status_code == 200
    d_clean = r_clean.json()
    assert "No Malicious Intent" in d_clean["attacker_intent"]["intent_title"]
    assert d_clean["attacker_intent"]["primary_vector"] == "✅ Safe"


def test_containment_assessment():
    # Direct endpoint test for containment
    actions = [
        "opened",
        "clicked_link",
        "opened_attachment",
        "entered_password",
        "entered_otp",
        "made_payment",
        "shared_sensitive_data"
    ]
    for action in actions:
        res = client.post("/containment/assess", json={"action_taken": action})
        assert res.status_code == 200
        d = res.json()
        assert d["action_taken"] == action
        assert "containment_playbook" in d
        assert len(d["containment_playbook"]["containment_steps"]) >= 3
        assert "incident_events" in d


def test_forgot_password_user_not_found():
    res = client.post("/auth/forgot-password", json={"email": "nonexistent.user.2026@mailshield.ai"})
    assert res.status_code == 404
    assert "No registered account found" in res.json()["detail"]


def test_forgot_password_and_reset_flow_success():
    uid = uuid.uuid4().hex[:6]
    test_email = f"analyst.reset.{uid}@mailshield.ai"
    test_name = "Alex Analyst"
    orig_pwd = "OriginalPassword123"
    new_pwd = "NewSecurePassword456!"

    # 1. Register user
    reg_res = client.post("/auth/register", json={
        "full_name": test_name,
        "email": test_email,
        "password": orig_pwd
    })
    assert reg_res.status_code == 200

    # 2. Login with original password works
    login_orig = client.post("/auth/login", json={"email": test_email, "password": orig_pwd})
    assert login_orig.status_code == 200
    assert "token" in login_orig.json()

    # 3. Request forgot password reset code
    forgot_res = client.post("/auth/forgot-password", json={"email": test_email})
    assert forgot_res.status_code == 200
    forgot_data = forgot_res.json()
    assert forgot_data["status"] == "success"
    assert "reset_code" in forgot_data
    assert len(forgot_data["reset_code"]) == 6
    reset_code = forgot_data["reset_code"]

    # 4. Perform password reset with code
    reset_res = client.post("/auth/reset-password", json={
        "email": test_email,
        "reset_code": reset_code,
        "new_password": new_pwd
    })
    assert reset_res.status_code == 200
    assert reset_res.json()["status"] == "success"

    # 5. Login with old password fails
    login_old = client.post("/auth/login", json={"email": test_email, "password": orig_pwd})
    assert login_old.status_code == 401

    # 6. Login with new password succeeds
    login_new = client.post("/auth/login", json={"email": test_email, "password": new_pwd})
    assert login_new.status_code == 200
    assert login_new.json()["user"]["email"] == test_email


def test_reset_password_invalid_code_rejected():
    uid = uuid.uuid4().hex[:6]
    test_email = f"analyst.code.{uid}@mailshield.ai"
    client.post("/auth/register", json={
        "full_name": "Test Code User",
        "email": test_email,
        "password": "ValidPassword123"
    })

    # Request code
    client.post("/auth/forgot-password", json={"email": test_email})

    # Submit invalid code
    bad_res = client.post("/auth/reset-password", json={
        "email": test_email,
        "reset_code": "000000",
        "new_password": "AnotherNewPassword123"
    })
    assert bad_res.status_code == 400
    assert "Invalid or expired verification code" in bad_res.json()["detail"]


def test_reset_password_short_password_rejected():
    uid = uuid.uuid4().hex[:6]
    test_email = f"analyst.short.{uid}@mailshield.ai"
    client.post("/auth/register", json={
        "full_name": "Short Pwd User",
        "email": test_email,
        "password": "ValidPassword123"
    })

    forgot_res = client.post("/auth/forgot-password", json={"email": test_email})
    reset_code = forgot_res.json()["reset_code"]

    # Submit too short password
    short_res = client.post("/auth/reset-password", json={
        "email": test_email,
        "reset_code": reset_code,
        "new_password": "123"
    })
    assert short_res.status_code == 400
    assert "at least 6 characters" in short_res.json()["detail"]


def test_gmail_status_disconnected_default():
    uid = uuid.uuid4().hex[:6]
    reg = client.post("/auth/register", json={
        "full_name": "Gmail Test User",
        "email": f"gmail.test.{uid}@mailshield.ai",
        "password": "Password123!"
    })
    token = reg.json()["token"]

    res = client.get("/api/gmail/status", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    data = res.json()
    assert data["connected"] is False


def test_gmail_messages_not_connected_error():
    uid = uuid.uuid4().hex[:6]
    reg = client.post("/auth/register", json={
        "full_name": "No Gmail User",
        "email": f"no.gmail.{uid}@mailshield.ai",
        "password": "Password123!"
    })
    token = reg.json()["token"]

    res = client.get("/api/gmail/messages", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 400
    assert "Gmail is not connected" in res.json()["detail"]


def test_gmail_mock_message_ingestion_and_analyze(monkeypatch):
    import app as app_module
    from database import update_user_google_tokens

    uid = uuid.uuid4().hex[:6]
    reg = client.post("/auth/register", json={
        "full_name": "Connected Analyst",
        "email": f"connected.analyst.{uid}@company.com",
        "password": "Password123!"
    })
    user_id = reg.json()["user"]["id"]
    token = reg.json()["token"]

    # Connect Google tokens in DB
    update_user_google_tokens(
        user_id=user_id,
        access_token="mock_google_access_token_12345",
        refresh_token="mock_refresh_token_67890",
        expiry="2099-01-01T00:00:00",
        gmail_email="connected.analyst@gmail.com"
    )

    # Mock list_gmail_messages
    def mock_list(access_token, max_results=20, query=None):
        return [
            {
                "id": "msg_phish_001",
                "thread_id": "thread_001",
                "sender": "Payroll Security <security-update@urgent-payroll-login.xyz>",
                "subject": "URGENT: Verify Your 6-Digit OTP & Password Now",
                "date": "Thu, 18 Sep 2026 10:15:00 GMT",
                "snippet": "Your account direct deposit is on hold. Enter your password and OTP code immediately.",
                "is_unread": True,
                "labels": ["INBOX", "UNREAD"]
            }
        ]

    # Mock get_gmail_message_detail
    def mock_detail(access_token, message_id):
        return {
            "message_id": message_id,
            "sender": "Payroll Security <security-update@urgent-payroll-login.xyz>",
            "to": "connected.analyst@company.com",
            "subject": "URGENT: Verify Your 6-Digit OTP & Password Now",
            "date": "Thu, 18 Sep 2026 10:15:00 GMT",
            "body": "Your payroll direct deposit is suspended. Please wire $5,000 and enter your OTP verification code at http://login-verify-portal.xyz/otp immediately. Attachment invoice.pdf is attached.",
            "raw_rfc822": "From: Payroll Security <security-update@urgent-payroll-login.xyz>\nSubject: URGENT: Verify Your 6-Digit OTP & Password Now\nDKIM-Signature: NONE\nReceived-SPF: FAIL\n\nYour payroll direct deposit is suspended. Please wire $5,000 and enter your OTP verification code at http://login-verify-portal.xyz/otp immediately. Attachment invoice.pdf is attached.",
            "attachments": [{"filename": "invoice.pdf", "mime_type": "application/pdf", "size": 15420}],
            "urls": ["http://login-verify-portal.xyz/otp"],
            "spf": "FAIL",
            "dkim": "NONE",
            "dmarc": "FAIL",
            "snippet": "Your account direct deposit is on hold..."
        }

    monkeypatch.setattr(app_module, "list_gmail_messages", mock_list)
    monkeypatch.setattr(app_module, "get_gmail_message_detail", mock_detail)

    # 1. Test status is now connected
    st_res = client.get("/api/gmail/status", headers={"Authorization": f"Bearer {token}"})
    assert st_res.status_code == 200
    assert st_res.json()["connected"] is True
    assert st_res.json()["gmail_email"] == "connected.analyst@gmail.com"

    # 2. Test messages list
    msg_res = client.get("/api/gmail/messages", headers={"Authorization": f"Bearer {token}"})
    assert msg_res.status_code == 200
    assert msg_res.json()["total"] == 1
    assert msg_res.json()["messages"][0]["id"] == "msg_phish_001"

    # 3. Test message detail
    det_res = client.get("/api/gmail/messages/msg_phish_001", headers={"Authorization": f"Bearer {token}"})
    assert det_res.status_code == 200
    assert len(det_res.json()["attachments"]) == 1

    # 4. Test automated ingestion and Attack Surface analysis
    ana_res = client.post("/api/gmail/messages/msg_phish_001/analyze", headers={"Authorization": f"Bearer {token}"})
    assert ana_res.status_code == 200
    ana_data = ana_res.json()
    assert ana_data["risk_level"] in ("Critical", "Elevated")
    assert "attack_surface_vectors" in ana_data
    assert len(ana_data["attack_surface_vectors"]) == 8

    # Verify detected attack vectors
    detected = [v["key"] for v in ana_data["attack_surface_vectors"] if v["detected"]]
    assert "otp" in detected
    assert "payment" in detected
    assert "link" in detected
    assert "what_can_happen" in ana_data
    assert "what_to_do_now" in ana_data
    assert len(ana_data["what_to_do_now"]) >= 3

    # 5. Test disconnect
    disc_res = client.post("/api/gmail/disconnect", headers={"Authorization": f"Bearer {token}"})
    assert disc_res.status_code == 200
    assert disc_res.json()["status"] == "success"

    # Verify status after disconnect
    st_after = client.get("/api/gmail/status", headers={"Authorization": f"Bearer {token}"})
    assert st_after.json()["connected"] is False


def test_link_intelligence_extraction_and_classification():
    # Test link intelligence with lookalike, shortener, IP, and benign HTTPS links
    sample = (
        "Hello User,\n"
        "1. Phishing portal: http://login-verify-auth-session-billing.net/invoice?id=99\n"
        "2. Shortened link: https://bit.ly/secure-account\n"
        "3. Direct IP link: http://192.168.1.100:8080/admin\n"
        "4. Legitimate link: https://docs.github.com/en/billing\n"
    )
    r = client.post("/predict", json={"raw_email": sample})
    assert r.status_code == 200
    data = r.json()
    assert "link_intelligence" in data
    links = data["link_intelligence"]
    assert len(links) >= 4

    statuses = {l["domain"]: l["status"] for l in links}
    reasons_map = {l["domain"]: l["reasons"] for l in links}

    # Lookalike should be SUSPICIOUS
    assert any("login-verify" in d for d in statuses)
    lookalike_dom = [d for d in statuses if "login-verify" in d][0]
    assert statuses[lookalike_dom] == "SUSPICIOUS"

    # Bitly should be WARNING (shortener)
    assert any("bit.ly" in d for d in statuses)
    assert statuses["bit.ly"] == "WARNING"

    # IP address should be SUSPICIOUS
    assert any("192.168.1.100" in d for d in statuses)
    ip_dom = [d for d in statuses if "192.168.1.100" in d][0]
    assert statuses[ip_dom] == "SUSPICIOUS"

    # GitHub should be CLEAN
    assert any("github.com" in d for d in statuses)
    gh_dom = [d for d in statuses if "github.com" in d][0]
    assert statuses[gh_dom] == "CLEAN"


def test_attachment_intelligence_double_extension_and_executables():
    sample = (
        "From: accounting@vendor-invoices.com\n"
        "Subject: Urgent Wire Invoice\n\n"
        "Please find your updated statement attached: Attachment: invoice_statement.pdf.exe\n"
        "Also see the backup script: Attachment: run_patch.vbs\n"
        "And clean file: Attachment: receipt.pdf\n"
    )
    r = client.post("/predict", json={"raw_email": sample})
    assert r.status_code == 200
    data = r.json()
    assert "attachment_intelligence" in data
    atts = data["attachment_intelligence"]
    assert len(atts) >= 3

    att_map = {a["filename"]: a for a in atts}
    assert "invoice_statement.pdf.exe" in att_map
    assert att_map["invoice_statement.pdf.exe"]["is_double_extension"] is True
    assert att_map["invoice_statement.pdf.exe"]["is_executable"] is True
    assert att_map["invoice_statement.pdf.exe"]["status"] == "HIGH_RISK"

    assert "run_patch.vbs" in att_map
    assert att_map["run_patch.vbs"]["is_script"] is True
    assert att_map["run_patch.vbs"]["status"] == "HIGH_RISK"

    assert "receipt.pdf" in att_map
    assert att_map["receipt.pdf"]["status"] == "SAFE"


def test_evidence_based_risk_score_calculation_and_breakdown():
    # High threat email with multiple evidence points
    sample = (
        "From: CEO <finance-update@ceo-payroll-sec.com>\n"
        "DKIM-Signature: NONE\n"
        "Received-SPF: FAIL\n"
        "Subject: URGENT: Wire Transfer Request $50,000\n\n"
        "Please execute an immediate wire transfer to http://login-verify-auth-session-billing.net/pay\n"
        "Attachment: details.zip\n"
    )
    r = client.post("/predict", json={"raw_email": sample})
    assert r.status_code == 200
    data = r.json()
    assert "evidence_risk_score" in data
    score_obj = data["evidence_risk_score"]
    assert score_obj["score"] >= 60
    assert score_obj["level"] in ("CRITICAL", "HIGH")
    assert len(score_obj["components"]) >= 3

    # Check categories in score breakdown
    categories = [c["category"] for c in score_obj["components"]]
    assert "sender" in categories or "url" in categories or "financial" in categories


def test_dynamic_protection_recommendations():
    # Phishing + wire + attachment sample
    sample = (
        "From: Security <alerts@spoofed-bank.xyz>\n"
        "Subject: Action Required: Reset Password & Wire Transfer\n\n"
        "Verify your password immediately at http://login-verify-auth-session-billing.net/reset\n"
        "Please transfer $2,000. Attachment: form.docx.exe\n"
    )
    r = client.post("/predict", json={"raw_email": sample})
    assert r.status_code == 200
    data = r.json()
    assert "protection_recommendations" in data
    recs = data["protection_recommendations"]
    assert len(recs) >= 3

    rec_categories = [rec["category"] for rec in recs]
    assert "link" in rec_categories
    assert "attachment" in rec_categories
    assert "credential" in rec_categories or "payment" in rec_categories
    for rec in recs:
        assert len(rec["actions"]) >= 2




