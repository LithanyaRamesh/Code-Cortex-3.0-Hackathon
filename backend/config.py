"""
MailShield AI - Central Configuration & Environment Manager
Handles loading environment variables from .env files, validation of Google OAuth credentials,
and centralized server parameters.
"""
import logging
import os
import re
from typing import Dict, Any, Optional

logger = logging.getLogger("mailshield.config")


def load_env():
    """Finds and loads .env files from backend or root directory."""
    candidates = [
        os.path.join(os.path.dirname(__file__), ".env"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
        os.path.join(os.getcwd(), "backend", ".env"),
        os.path.join(os.getcwd(), ".env")
    ]
    
    loaded = False
    for path in candidates:
        if os.path.isfile(path):
            try:
                import dotenv
                dotenv.load_dotenv(path, override=False)
                loaded = True
            except ImportError:
                pass
            
            # Pure Python fallback parser
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if line.startswith("export "):
                            line = line[7:].strip()
                        if "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip("'\"")
                            if k and k not in os.environ:
                                os.environ[k] = v
                loaded = True
            except Exception as e:
                logger.warning(f"Error reading .env at {path}: {e}")
    
    return loaded


# Run load on initial import
load_env()


def get_env(key: str, default: str = "") -> str:
    """Gets an environment variable, refreshing from .env if unset."""
    val = os.environ.get(key, "").strip()
    if not val:
        load_env()
        val = os.environ.get(key, "").strip()
    return val or default


def get_google_client_id() -> str:
    return get_env("GOOGLE_CLIENT_ID", "")


def get_google_client_secret() -> str:
    return get_env("GOOGLE_CLIENT_SECRET", "")


def get_google_redirect_uri() -> str:
    return get_env("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback")


def get_frontend_url() -> str:
    return get_env("FRONTEND_URL", "http://localhost:3000")


def get_secret_key() -> str:
    return get_env("MAILSHIELD_SECRET_KEY", "mailshield-ai-cybersecurity-secret-key-2026")


def is_google_oauth_configured() -> bool:
    """Checks whether Google OAuth credentials are appropriately configured."""
    cid = get_google_client_id()
    sec = get_google_client_secret()
    if not cid or not sec:
        return False
    if cid.startswith("YOUR_") or sec.startswith("YOUR_") or "example" in cid.lower():
        return False
    return True


def get_google_oauth_info() -> Dict[str, Any]:
    configured = is_google_oauth_configured()
    return {
        "configured": configured,
        "client_id": get_google_client_id() if configured else None,
        "redirect_uri": get_google_redirect_uri(),
        "frontend_url": get_frontend_url(),
        "message": "Google OAuth is ready." if configured else (
            "Google OAuth is not configured on this server. "
            "To enable Google Sign-In, please set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET environment variables in your backend environment."
        )
    }
