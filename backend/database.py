import json
import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any

DB_PATH = os.environ.get(
    "MAILSHIELD_DB_PATH",
    os.path.join(os.path.dirname(__file__), "mailshield.db")
)


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL COLLATE NOCASE,
        full_name TEXT NOT NULL,
        password_hash TEXT NOT NULL,
        salt TEXT NOT NULL,
        auth_provider TEXT NOT NULL DEFAULT 'local',
        created_at TEXT NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS scans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        scan_id TEXT NOT NULL UNIQUE,
        subject_preview TEXT,
        verdict TEXT NOT NULL,
        risk_level TEXT NOT NULL,
        confidence REAL NOT NULL,
        model_probability_spam REAL NOT NULL,
        explanation TEXT,
        indicators_json TEXT,
        spf TEXT,
        dkim TEXT,
        dmarc TEXT,
        suspicious_links INTEGER DEFAULT 0,
        raw_snippet TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    """)
    
    conn.commit()
    conn.close()


def create_user(email: str, full_name: str, password_hash: str, salt: str, auth_provider: str = "local") -> Dict[str, Any]:
    conn = get_db_connection()
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()
    try:
        cur.execute(
            "INSERT INTO users (email, full_name, password_hash, salt, auth_provider, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (email.strip().lower(), full_name.strip(), password_hash, salt, auth_provider, now)
        )
        conn.commit()
        user_id = cur.lastrowid
        return {
            "id": user_id,
            "email": email.strip().lower(),
            "full_name": full_name.strip(),
            "auth_provider": auth_provider,
            "created_at": now
        }
    finally:
        conn.close()


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT * FROM users WHERE email = ? COLLATE NOCASE", (email.strip().lower(),))
        row = cur.fetchone()
        if row:
            return dict(row)
        return None
    finally:
        conn.close()


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cur.fetchone()
        if row:
            return dict(row)
        return None
    finally:
        conn.close()


def save_scan(scan_data: Dict[str, Any], user_id: Optional[int] = None) -> Dict[str, Any]:
    conn = get_db_connection()
    cur = conn.cursor()
    now = scan_data.get("timestamp") or datetime.utcnow().isoformat()
    indicators_json = json.dumps(scan_data.get("indicators", []))
    
    try:
        cur.execute("""
        INSERT INTO scans (
            user_id, scan_id, subject_preview, verdict, risk_level,
            confidence, model_probability_spam, explanation, indicators_json,
            spf, dkim, dmarc, suspicious_links, raw_snippet, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            scan_data["scan_id"],
            scan_data.get("subject_preview", ""),
            scan_data["verdict"],
            scan_data["risk_level"],
            scan_data["confidence"],
            scan_data["model_probability_spam"],
            scan_data.get("explanation", ""),
            indicators_json,
            scan_data.get("spf", "UNKNOWN"),
            scan_data.get("dkim", "UNKNOWN"),
            scan_data.get("dmarc", "UNKNOWN"),
            scan_data.get("suspicious_links", 0),
            scan_data.get("raw_snippet", ""),
            now
        ))
        conn.commit()
        return scan_data
    finally:
        conn.close()


def get_scans(user_id: Optional[int] = None, limit: int = 50) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        if user_id is not None:
            cur.execute("SELECT * FROM scans WHERE user_id = ? ORDER BY id DESC LIMIT ?", (user_id, limit))
        else:
            cur.execute("SELECT * FROM scans ORDER BY id DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
        results = []
        for r in rows:
            d = dict(r)
            try:
                d["indicators"] = json.loads(d.get("indicators_json") or "[]")
            except Exception:
                d["indicators"] = []
            results.append(d)
        return results
    finally:
        conn.close()


def get_scan_by_id(scan_id: str, user_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        if user_id is not None:
            cur.execute("SELECT * FROM scans WHERE scan_id = ? AND user_id = ?", (scan_id, user_id))
        else:
            cur.execute("SELECT * FROM scans WHERE scan_id = ?", (scan_id,))
        row = cur.fetchone()
        if not row:
            return None
        d = dict(row)
        try:
            d["indicators"] = json.loads(d.get("indicators_json") or "[]")
        except Exception:
            d["indicators"] = []
        return d
    finally:
        conn.close()


def get_stats(user_id: Optional[int] = None) -> Dict[str, Any]:
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        if user_id is not None:
            cur.execute("SELECT * FROM scans WHERE user_id = ?", (user_id,))
        else:
            cur.execute("SELECT * FROM scans")
        rows = cur.fetchall()
        
        total = len(rows)
        if total == 0:
            return {
                "total_analyzed": 0,
                "threats_count": 0,
                "legitimate_count": 0,
                "avg_confidence": 0.0,
                "legit_percent": 0.0,
                "spam_percent": 0.0,
                "phishing_percent": 0.0,
                "attack_vectors": {
                    "bec_count": 0,
                    "phishing_count": 0,
                    "suspicious_link_count": 0
                },
                "daily_volume": [
                    {"day": "Mon", "legit": 0, "spam": 0, "phishing": 0, "total": 0},
                    {"day": "Tue", "legit": 0, "spam": 0, "phishing": 0, "total": 0},
                    {"day": "Wed", "legit": 0, "spam": 0, "phishing": 0, "total": 0},
                    {"day": "Thu", "legit": 0, "spam": 0, "phishing": 0, "total": 0},
                    {"day": "Fri", "legit": 0, "spam": 0, "phishing": 0, "total": 0},
                    {"day": "Sat", "legit": 0, "spam": 0, "phishing": 0, "total": 0},
                    {"day": "Sun", "legit": 0, "spam": 0, "phishing": 0, "total": 0},
                ],
                "has_data": False
            }

        threats = 0
        legit = 0
        phishing = 0
        spam = 0
        conf_sum = 0.0
        bec_count = 0
        phish_portal_count = 0
        susp_link_total = 0

        days_map = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        daily_stats = [{"day": d, "legit": 0, "spam": 0, "phishing": 0, "total": 0} for d in day_names]

        for r in rows:
            risk = r["risk_level"]
            verdict = r["verdict"].upper()
            conf = float(r["confidence"])
            conf_sum += conf
            susp_links = int(r["suspicious_links"] or 0)
            if susp_links > 0:
                susp_link_total += susp_links

            # Risk classification
            if risk == "Low" or "LEGITIMATE" in verdict:
                legit += 1
                cat = "legit"
            elif "CRITICAL" in verdict or "PHISHING" in verdict:
                phishing += 1
                threats += 1
                cat = "phishing"
                phish_portal_count += 1
            else:
                spam += 1
                threats += 1
                cat = "spam"

            # Check indicators for BEC or specific vectors
            ind_raw = r["indicators_json"] or ""
            if "BEC" in ind_raw or "Urgency" in ind_raw or "Wire" in ind_raw or "Financial" in ind_raw:
                bec_count += 1

            # Day classification
            try:
                dt = datetime.fromisoformat(r["created_at"])
                day_name = dt.strftime("%a")
                if day_name in days_map:
                    idx = days_map[day_name]
                    daily_stats[idx][cat] += 1
                    daily_stats[idx]["total"] += 1
            except Exception:
                pass

        return {
            "total_analyzed": total,
            "threats_count": threats,
            "legitimate_count": legit,
            "avg_confidence": round(conf_sum / total, 1),
            "legit_percent": round((legit / total) * 100, 1),
            "spam_percent": round((spam / total) * 100, 1),
            "phishing_percent": round((phishing / total) * 100, 1),
            "attack_vectors": {
                "bec_count": bec_count,
                "phishing_count": phish_portal_count,
                "suspicious_link_count": susp_link_total
            },
            "daily_volume": daily_stats,
            "has_data": True
        }
    finally:
        conn.close()
