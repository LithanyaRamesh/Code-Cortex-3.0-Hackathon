# MailShield AI

Enterprise email spam & phishing threat defense platform built with a high-fidelity Figma-inspired cybersecurity UI, backed by a **production-grade trained Machine Learning classifier**, **SQLite persistence**, and **secure authentication**.

## Core System Architecture & Features

- **Trained Machine Learning Engine**: `backend/train_model.py` trains a TF-IDF + Logistic Regression model on 5,695 deduplicated emails (5,728 raw with 33 duplicates removed: 4,327 legitimate, 1,368 spam) using an 80/20 stratified split.
  - **Accuracy: 98.51%**
  - **Spam Precision: 100.0%**
  - **Spam Recall: 93.80%**
  - **Spam F1-Score: 96.80%**
  - Stored in `backend/model/classifier.joblib`, `backend/model/vectorizer.joblib`, and `backend/model/metrics.json`.
- **Authentication & User Management**:
  - Registration with Full Name, Email, and Password.
  - Secure password hashing using PBKDF2-HMAC-SHA256 (120,000 iterations) with cryptographic salts.
  - JWT session token generation and verification.
  - Protected routes and profile personalization with user initials avatar.
  - Google OAuth integration support with environment variable validation (`GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`). Shows clear configuration messaging when not configured.
- **Persistence & Telemetry**:
  - SQLite database (`backend/mailshield.db`) storing registered accounts and real scan logs.
  - Zero fabricated stats: dynamic telemetry calculates real percentages, threat counts, attack vectors (BEC, Phishing Portals, Suspicious Links), and weekly volume from stored scan records.
  - Informative empty states when no emails have been analyzed.
- **Deep Threat Analysis**:
  - Combines trained ML classification with SPF, DKIM, DMARC forensic checks, and lookalike/high-risk domain detection.
  - Instant inspection via `POST /analyze` and `POST /predict`.
  - Stored scan history with single-click inspection report drilldown.
  - Export scan logs to CSV.

## Getting Started

### 1. Backend Setup
```bash
cd backend
pip install -r requirements.txt
python train_model.py              # Train model and generate metrics
uvicorn app:app --reload --port 8000
```

### 2. Frontend Launch
Open `frontend/index.html` in your web browser (or serve with any static HTTP server).

### 3. Google OAuth Configuration (Optional)
To enable Google Sign-In, set the following environment variables:
```bash
export GOOGLE_CLIENT_ID="your-google-client-id"
export GOOGLE_CLIENT_SECRET="your-google-client-secret"
```

## Running Automated Tests

```bash
cd backend
python -m pytest
```

Test suites include:
- `tests/test_auth.py`: Registration, password hashing, invalid login rejection, duplicate email checks, and Google OAuth status.
- `tests/test_model.py`: Model loading, accuracy thresholds, and prediction sanity.
- `tests/test_api.py`: `/health`, `/metrics`, `/analyze`, `/predict`, input validation, history, and stats endpoints.
- `tests/test_e2e_flow.py`: Full end-to-end integration workflow spanning all 16 verification requirements.

## Project Structure

```
mailshield-ai/
├── frontend/
│   └── index.html          # Dark cybersecurity UI (Dashboard, Analyzer, Reports, Settings, Auth)
├── backend/
│   ├── app.py               # FastAPI service with auth, ML inference, and telemetry endpoints
│   ├── auth.py              # PBKDF2 hashing, JWT generation/verification, Google OAuth helpers
│   ├── database.py          # SQLite database schema, CRUD, and telemetry aggregators
│   ├── train_model.py       # Model training with deduplication & metrics export
│   ├── requirements.txt
│   ├── data/
│   │   └── emails.csv       # Corpus dataset
│   ├── model/
│   │   ├── vectorizer.joblib
│   │   ├── classifier.joblib
│   │   └── metrics.json     # Test benchmark metrics
│   └── tests/
│       ├── test_api.py
│       ├── test_auth.py
│       ├── test_e2e_flow.py
│       └── test_model.py
└── README.md
```
