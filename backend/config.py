"""
MailShield AI - Central Configuration & Environment Manager
Handles loading environment variables from .env files, validation of Google OAuth credentials,
safe diagnostics, and centralized server parameters.
"""
import logging
import os
import sys
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger("mailshield.config")

_LAST_LOADED_ENV_PATH: Optional[str] = None


def find_env_file() -> Optional[str]:
    """Scans potential candidate paths for .env files."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    cwd = os.path.abspath(os.getcwd())

    candidates = [
        os.path.join(base_dir, ".env"),
        os.path.join(parent_dir, ".env"),
        os.path.join(cwd, "backend", ".env"),
        os.path.join(cwd, ".env")
    ]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_candidates = []
    for c in candidates:
        norm = os.path.normpath(c)
        if norm not in seen:
            seen.add(norm)
            unique_candidates.append(norm)

    for path in unique_candidates:
        if os.path.isfile(path):
            return path
    return None


def parse_env_file(filepath: str) -> Dict[str, str]:
    """Robust parser for .env files without external dependencies."""
    parsed = {}
    if not os.path.isfile(filepath):
        return parsed
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[7:].strip()
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip()
                    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                        val = val[1:-1]
                    if key:
                        parsed[key] = val
    except Exception as e:
        logger.warning(f"Error parsing .env file at {filepath}: {e}")
    return parsed


def load_env(force_reload: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Finds and loads .env files into os.environ.
    If force_reload is True or a key has an empty value in os.environ,
    the .env value will override it.
    """
    global _LAST_LOADED_ENV_PATH
    env_path = find_env_file()
    if not env_path:
        return False, None

    _LAST_LOADED_ENV_PATH = env_path
    
    # Try python-dotenv first if available
    try:
        import dotenv
        dotenv.load_dotenv(env_path, override=force_reload)
    except ImportError:
        pass

    # Apply parsed values directly to guarantee consistency
    parsed = parse_env_file(env_path)
    for k, v in parsed.items():
        if force_reload or k not in os.environ or not os.environ.get(k, "").strip():
            if v:
                os.environ[k] = v

    return True, env_path


# Run load on initial module import
load_env(force_reload=True)


def get_env(key: str, default: str = "") -> str:
    """Gets an environment variable, refreshing from .env if empty."""
    val = os.environ.get(key, "").strip()
    if not val:
        load_env(force_reload=True)
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


def get_missing_oauth_variables() -> List[str]:
    """Identifies which required Google OAuth environment variables are missing."""
    missing = []
    cid = get_google_client_id()
    secret = get_google_client_secret()

    if not cid or not cid.strip():
        missing.append("GOOGLE_CLIENT_ID")
    if not secret or not secret.strip():
        missing.append("GOOGLE_CLIENT_SECRET")
    return missing


def is_google_oauth_configured() -> bool:
    """Checks whether Google OAuth credentials are appropriately configured."""
    return len(get_missing_oauth_variables()) == 0


def get_startup_diagnostics() -> Dict[str, Any]:
    """Returns safe diagnostics status for configuration variables."""
    cid = get_google_client_id()
    sec = get_google_client_secret()
    redirect_uri = get_google_redirect_uri()
    frontend_url = get_frontend_url()
    missing = get_missing_oauth_variables()
    env_file = find_env_file()

    return {
        "GOOGLE_CLIENT_ID": "configured" if ("GOOGLE_CLIENT_ID" not in missing and cid) else "missing",
        "GOOGLE_CLIENT_SECRET": "configured" if ("GOOGLE_CLIENT_SECRET" not in missing and sec) else "missing",
        "GOOGLE_REDIRECT_URI": "configured" if redirect_uri else "missing",
        "FRONTEND_URL": "configured" if frontend_url else "missing",
        "env_file_loaded": env_file or "Not Found (Using default/system env)",
        "oauth_ready": len(missing) == 0,
        "missing_variables": missing
    }


def print_startup_diagnostics():
    """Prints startup configuration diagnostics safely without exposing secrets."""
    diag = get_startup_diagnostics()
    print("=" * 64, flush=True)
    print(" MailShield AI - Configuration & OAuth Diagnostics", flush=True)
    print("=" * 64, flush=True)
    print(f" GOOGLE_CLIENT_ID:      {diag['GOOGLE_CLIENT_ID']}", flush=True)
    print(f" GOOGLE_CLIENT_SECRET:  {diag['GOOGLE_CLIENT_SECRET']}", flush=True)
    print(f" GOOGLE_REDIRECT_URI:   {diag['GOOGLE_REDIRECT_URI']} ({get_google_redirect_uri()})", flush=True)
    print(f" FRONTEND_URL:          {diag['FRONTEND_URL']} ({get_frontend_url()})", flush=True)
    print("-" * 64, flush=True)
    print(f" Config File Source:    {diag['env_file_loaded']}", flush=True)
    if diag["oauth_ready"]:
        print(" Google OAuth Status:   READY (All credentials configured)", flush=True)
    else:
        print(f" Google OAuth Status:   NOT CONFIGURED (Missing: {', '.join(diag['missing_variables'])})", flush=True)
    print("=" * 64, flush=True)


def get_google_oauth_info() -> Dict[str, Any]:
    """Returns OAuth status payload with precise instructions if unconfigured."""
    diag = get_startup_diagnostics()
    configured = diag["oauth_ready"]
    missing = diag["missing_variables"]
    
    if configured:
        msg = "Google OAuth is ready."
    else:
        missing_str = ", ".join(missing) if missing else "credentials"
        msg = (
            f"Google OAuth is not configured on this server. "
            f"Missing required environment variable(s): {missing_str}. "
            f"Please configure them in backend/.env to enable Google Sign-In and Gmail Ingestion."
        )

    return {
        "configured": configured,
        "client_id": get_google_client_id() if configured else None,
        "redirect_uri": get_google_redirect_uri(),
        "frontend_url": get_frontend_url(),
        "missing_variables": missing,
        "env_file": diag["env_file_loaded"],
        "message": msg
    }
