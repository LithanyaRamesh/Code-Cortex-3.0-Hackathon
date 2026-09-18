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
def setup_db():
    init_db()


def test_register_and_login_success():
    email = f"analyst_{uuid.uuid4().hex[:8]}@cyberdefense.io"
    # Register
    reg_resp = client.post("/auth/register", json={
        "full_name": "Alex Mercer",
        "email": email,
        "password": "SecurePassword123!"
    })
    assert reg_resp.status_code == 200
    reg_data = reg_resp.json()
    assert "token" in reg_data
    assert reg_data["user"]["email"] == email

    # Valid Login
    login_resp = client.post("/auth/login", json={
        "email": email,
        "password": "SecurePassword123!"
    })
    assert login_resp.status_code == 200
    data = login_resp.json()
    assert "token" in data
    assert data["user"]["email"] == email
    assert data["user"]["full_name"] == "Alex Mercer"

    # Profile me endpoint
    token = data["token"]
    me_resp = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_resp.status_code == 200
    assert me_resp.json()["email"] == email


def test_invalid_login_rejected():
    resp = client.post("/auth/login", json={
        "email": "nonexistent_user@invalid.com",
        "password": "WrongPassword999"
    })
    assert resp.status_code == 401
    assert "detail" in resp.json()


def test_duplicate_registration_rejected():
    email = f"lead_{uuid.uuid4().hex[:8]}@enterprise.com"
    r1 = client.post("/auth/register", json={
        "full_name": "SOC Lead",
        "email": email,
        "password": "Password1234!"
    })
    assert r1.status_code == 200

    # Duplicate
    r2 = client.post("/auth/register", json={
        "full_name": "Another Name",
        "email": email,
        "password": "AnotherPassword!"
    })
    assert r2.status_code == 400
    assert "already exists" in r2.json()["detail"].lower()


def test_google_oauth_status_unconfigured():
    resp = client.get("/auth/google/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "configured" in data
    assert "message" in data

    # Attempting to login via Google when unconfigured should return 501
    if not data["configured"]:
        g_login = client.post("/auth/google", json={"credential": "mock_google_token"})
        assert g_login.status_code == 501
        assert "not configured" in g_login.json()["detail"].lower()
