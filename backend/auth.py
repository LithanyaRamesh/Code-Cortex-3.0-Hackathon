import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Dict, Optional, Tuple, Any

SECRET_KEY = os.environ.get("MAILSHIELD_SECRET_KEY", "mailshield-ai-cybersecurity-secret-key-2026")
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.environ.get("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback")


def hash_password(password: str) -> Tuple[str, str]:
    """Generates cryptographic salt and PBKDF2-HMAC-SHA256 hash."""
    salt = secrets.token_bytes(16)
    salt_hex = salt.hex()
    pwd_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        120000
    )
    return pwd_hash.hex(), salt_hex


def verify_password(password: str, password_hash: str, salt_hex: str) -> bool:
    """Verifies a password against the stored hash and salt."""
    try:
        salt = bytes.fromhex(salt_hex)
        computed_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            120000
        )
        return hmac.compare_digest(computed_hash.hex(), password_hash)
    except Exception:
        return False


def _b64_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _b64_decode(data: str) -> bytes:
    padding = "=" * ((4 - len(data) % 4) % 4)
    return base64.urlsafe_b64decode(data + padding)


def create_jwt_token(payload: Dict[str, Any], expires_seconds: int = 86400 * 7) -> str:
    """Creates an HMAC-SHA256 signed JWT token."""
    header = {"alg": "HS256", "typ": "JWT"}
    body = {
        **payload,
        "exp": int(time.time()) + expires_seconds,
        "iat": int(time.time())
    }
    
    header_b64 = _b64_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = _b64_encode(json.dumps(body, separators=(",", ":")).encode("utf-8"))
    
    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    signature = hmac.new(SECRET_KEY.encode("utf-8"), signing_input, hashlib.sha256).digest()
    sig_b64 = _b64_encode(signature)
    
    return f"{header_b64}.{payload_b64}.{sig_b64}"


def decode_jwt_token(token: str) -> Optional[Dict[str, Any]]:
    """Decodes and validates an HMAC-SHA256 signed JWT token."""
    try:
        parts = token.strip().split(".")
        if len(parts) != 3:
            return None
        header_b64, payload_b64, sig_b64 = parts
        
        signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
        expected_sig = hmac.new(SECRET_KEY.encode("utf-8"), signing_input, hashlib.sha256).digest()
        actual_sig = _b64_decode(sig_b64)
        
        if not hmac.compare_digest(expected_sig, actual_sig):
            return None
        
        payload_bytes = _b64_decode(payload_b64)
        payload = json.loads(payload_bytes.decode("utf-8"))
        
        # Expiration check
        if payload.get("exp") and time.time() > payload["exp"]:
            return None
        
        return payload
    except Exception:
        return None


def is_google_oauth_configured() -> bool:
    """Checks if Google OAuth credentials are configured via environment variables."""
    cid = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
    sec = os.environ.get("GOOGLE_CLIENT_SECRET", "").strip()
    return bool(cid and sec and not cid.startswith("YOUR_") and not sec.startswith("YOUR_"))


def get_google_oauth_info() -> Dict[str, Any]:
    configured = is_google_oauth_configured()
    return {
        "configured": configured,
        "client_id": os.environ.get("GOOGLE_CLIENT_ID", "").strip() if configured else None,
        "message": "Google OAuth is ready." if configured else (
            "Google OAuth is not configured on this server. "
            "To enable Google Sign-In, please set the GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET environment variables in your backend environment."
        )
    }
