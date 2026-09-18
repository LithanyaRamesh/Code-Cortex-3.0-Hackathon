"""
MailShield AI - Gmail Ingestion and OAuth Service
Handles Google OAuth 2.0 flow, token refreshes, message listing, full MIME decoding,
attachment/link extraction, and synthesis for downstream AI/ML and Attack Surface detection.
"""
import base64
import email
from email import policy
import html
import json
import logging
import os
import re
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from config import (
    get_google_client_id, get_google_client_secret,
    get_google_redirect_uri, get_frontend_url
)

logger = logging.getLogger("mailshield.gmail")

GMAIL_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/gmail.readonly"
]

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"
GMAIL_API_BASE = "https://gmail.googleapis.com/gmail/v1/users/me"


def get_google_auth_url(state: str = "") -> str:
    """Constructs the Google OAuth 2.0 authorization URL requesting Gmail read-only access."""
    cid = get_google_client_id()
    redirect_uri = get_google_redirect_uri()
    
    params = {
        "client_id": cid,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(GMAIL_SCOPES),
        "access_type": "offline",
        "prompt": "consent",
        "state": state
    }
    return f"{GOOGLE_AUTH_URL}?{urllib.parse.urlencode(params)}"


def exchange_code_for_tokens(code: str) -> Dict[str, Any]:
    """Exchanges an authorization code for access and refresh tokens."""
    cid = get_google_client_id()
    secret = get_google_client_secret()
    redirect_uri = get_google_redirect_uri()
    
    data = urllib.parse.urlencode({
        "code": code,
        "client_id": cid,
        "client_secret": secret,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code"
    }).encode("utf-8")
    
    req = urllib.request.Request(
        GOOGLE_TOKEN_URL,
        data=data,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "MailShield-AI/2.1"
        }
    )
    
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_msg = ""
        try:
            raw_err = e.read().decode("utf-8")
            err_json = json.loads(raw_err)
            err_msg = err_json.get("error_description") or err_json.get("error") or raw_err
        except Exception:
            err_msg = str(e)
        logger.error(f"Google token exchange HTTP {e.code} error: {err_msg}")
        raise RuntimeError(f"Google OAuth token exchange failed ({e.code}): {err_msg}") from e
    except Exception as e:
        logger.error(f"Google token exchange network error: {e}")
        raise


def refresh_access_token(refresh_token: str) -> Dict[str, Any]:
    """Uses the stored refresh token to obtain a fresh access token."""
    cid = get_google_client_id()
    secret = get_google_client_secret()
    
    data = urllib.parse.urlencode({
        "refresh_token": refresh_token,
        "client_id": cid,
        "client_secret": secret,
        "grant_type": "refresh_token"
    }).encode("utf-8")
    
    req = urllib.request.Request(
        GOOGLE_TOKEN_URL,
        data=data,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "MailShield-AI/2.1"
        }
    )
    
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_msg = ""
        try:
            raw_err = e.read().decode("utf-8")
            err_json = json.loads(raw_err)
            err_msg = err_json.get("error_description") or err_json.get("error") or raw_err
        except Exception:
            err_msg = str(e)
        logger.error(f"Google token refresh HTTP {e.code} error: {err_msg}")
        raise RuntimeError(f"Google OAuth token refresh failed ({e.code}): {err_msg}") from e
    except Exception as e:
        logger.error(f"Google token refresh network error: {e}")
        raise


def get_google_user_info(access_token: str) -> Dict[str, Any]:
    """Fetches user profile information from Google."""
    req = urllib.request.Request(
        GOOGLE_USERINFO_URL,
        headers={
            "Authorization": f"Bearer {access_token}",
            "User-Agent": "MailShield-AI/2.1"
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_msg = ""
        try:
            raw_err = e.read().decode("utf-8")
            err_json = json.loads(raw_err)
            err_msg = err_json.get("error_description") or err_json.get("error") or raw_err
        except Exception:
            err_msg = str(e)
        logger.error(f"Google userinfo fetch HTTP {e.code} error: {err_msg}")
        raise RuntimeError(f"Google userinfo fetch failed ({e.code}): {err_msg}") from e
    except Exception as e:
        logger.error(f"Google userinfo network error: {e}")
        raise


def _clean_base64_decode(data_str: str) -> str:
    """Decodes standard or URL-safe base64 encoded strings."""
    if not data_str:
        return ""
    try:
        clean_str = data_str.replace("-", "+").replace("_", "/")
        padding = "=" * ((4 - len(clean_str) % 4) % 4)
        raw_bytes = base64.b64decode(clean_str + padding)
        try:
            return raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return raw_bytes.decode("latin-1", errors="replace")
    except Exception as e:
        logger.warning(f"Error decoding base64 data: {e}")
        return ""


def _strip_html(html_text: str) -> str:
    """Converts HTML markup into readable plain text."""
    if not html_text:
        return ""
    text = re.sub(r"<style[^>]*>.*?</style>", "", html_text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_parts(part: Dict[str, Any], plain_parts: List[str], html_parts: List[str], attachments: List[Dict[str, Any]]):
    """Recursively traverses Gmail MIME payload parts to extract body contents and attachments."""
    mime_type = part.get("mimeType", "").lower()
    filename = part.get("filename", "")
    body_data = part.get("body", {})
    
    # Check for attachment
    if filename:
        attachments.append({
            "filename": filename,
            "mime_type": mime_type,
            "size": body_data.get("size", 0),
            "attachment_id": body_data.get("attachmentId", "")
        })
    
    # Extract body content
    raw_data = body_data.get("data", "")
    if raw_data:
        decoded = _clean_base64_decode(raw_data)
        if "text/plain" in mime_type:
            plain_parts.append(decoded)
        elif "text/html" in mime_type:
            html_parts.append(decoded)
    
    # Traverse subparts
    subparts = part.get("parts", [])
    for sub in subparts:
        _extract_parts(sub, plain_parts, html_parts, attachments)


def list_gmail_messages(access_token: str, max_results: int = 20, query: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetches message list and summaries from Gmail API."""
    params = {"maxResults": min(max_results, 50)}
    if query:
        params["q"] = query
    
    url = f"{GMAIL_API_BASE}/messages?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {access_token}"}
    )
    
    with urllib.request.urlopen(req, timeout=20) as response:
        data = json.loads(response.read().decode("utf-8"))
    
    messages = data.get("messages", [])
    if not messages:
        return []
    
    summaries = []
    # Fetch message headers for each message (limit to top results)
    for m in messages[:max_results]:
        msg_id = m.get("id")
        try:
            msg_url = f"{GMAIL_API_BASE}/messages/{msg_id}?format=metadata&metadataHeaders=From&metadataHeaders=Subject&metadataHeaders=Date&metadataHeaders=To"
            m_req = urllib.request.Request(
                msg_url,
                headers={"Authorization": f"Bearer {access_token}"}
            )
            with urllib.request.urlopen(m_req, timeout=10) as m_resp:
                m_data = json.loads(m_resp.read().decode("utf-8"))
            
            headers = {h["name"].lower(): h["value"] for h in m_data.get("payload", {}).get("headers", [])}
            
            sender = headers.get("from", "Unknown Sender")
            subject = headers.get("subject", "(No Subject)")
            date_str = headers.get("date", "")
            snippet = m_data.get("snippet", "")
            label_ids = m_data.get("labelIds", [])
            is_unread = "UNREAD" in label_ids
            
            summaries.append({
                "id": msg_id,
                "thread_id": m_data.get("threadId"),
                "sender": sender,
                "subject": subject,
                "date": date_str,
                "snippet": snippet,
                "is_unread": is_unread,
                "labels": label_ids
            })
        except Exception as e:
            logger.warning(f"Error fetching metadata for message {msg_id}: {e}")
            summaries.append({
                "id": msg_id,
                "sender": "Unknown Sender",
                "subject": f"Message {msg_id}",
                "date": "",
                "snippet": "",
                "is_unread": False,
                "labels": []
            })
    
    return summaries


def get_gmail_message_detail(access_token: str, message_id: str) -> Dict[str, Any]:
    """
    Fetches full message content, decodes MIME body, extracts attachments & links,
    and constructs a raw email representation ready for AI/ML and Attack Surface analysis.
    """
    url = f"{GMAIL_API_BASE}/messages/{message_id}?format=full"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {access_token}"}
    )
    
    with urllib.request.urlopen(req, timeout=20) as response:
        msg = json.loads(response.read().decode("utf-8"))
    
    payload = msg.get("payload", {})
    headers_list = payload.get("headers", [])
    headers = {h["name"].lower(): h["value"] for h in headers_list}
    
    # Forensic headers
    sender = headers.get("from", "")
    to_addr = headers.get("to", "")
    subject = headers.get("subject", "(No Subject)")
    date_str = headers.get("date", "")
    dkim_sig = headers.get("dkim-signature", "")
    auth_results = headers.get("authentication-results", "")
    received_spf = headers.get("received-spf", "")
    
    # Extract body & attachments
    plain_parts = []
    html_parts = []
    attachments = []
    
    _extract_parts(payload, plain_parts, html_parts, attachments)
    
    body_text = ""
    if plain_parts:
        body_text = "\n\n".join(plain_parts).strip()
    elif html_parts:
        body_text = _strip_html("\n\n".join(html_parts))
    else:
        body_text = msg.get("snippet", "")
    
    # Extract URLs from body and HTML
    url_pattern = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
    extracted_urls = list(set(url_pattern.findall(body_text + " " + " ".join(html_parts))))
    
    # Evaluate SPF/DKIM/DMARC from headers
    spf = "PASS" if ("spf=pass" in auth_results.lower() or "pass" in received_spf.lower()) else ("FAIL" if ("spf=fail" in auth_results.lower() or "fail" in received_spf.lower()) else "UNKNOWN")
    dkim = "PASS" if ("dkim=pass" in auth_results.lower() or len(dkim_sig) > 10) else ("FAIL" if "dkim=fail" in auth_results.lower() else "NONE")
    dmarc = "PASS" if "dmarc=pass" in auth_results.lower() else ("FAIL" if "dmarc=fail" in auth_results.lower() else "UNKNOWN")
    
    # Synthesize standard raw RFC 822 format for model ingestion
    raw_lines = [
        f"From: {sender}",
        f"To: {to_addr}",
        f"Subject: {subject}",
        f"Date: {date_str}"
    ]
    if dkim != "NONE":
        raw_lines.append(f"DKIM-Signature: {dkim}")
    if spf != "UNKNOWN":
        raw_lines.append(f"Received-SPF: {spf}")
    if dmarc != "UNKNOWN":
        raw_lines.append(f"DMARC-Status: {dmarc}")
    
    raw_lines.append("")
    raw_lines.append(body_text)
    
    if attachments:
        raw_lines.append("")
        raw_lines.append("--- Attachments ---")
        for att in attachments:
            raw_lines.append(f"Attachment: {att.get('filename')} ({att.get('mime_type')}, {att.get('size')} bytes)")
    
    raw_rfc822 = "\n".join(raw_lines)
    
    return {
        "message_id": message_id,
        "sender": sender,
        "to": to_addr,
        "subject": subject,
        "date": date_str,
        "body": body_text,
        "raw_rfc822": raw_rfc822,
        "attachments": attachments,
        "urls": extracted_urls,
        "spf": spf,
        "dkim": dkim,
        "dmarc": dmarc,
        "snippet": msg.get("snippet", "")
    }
