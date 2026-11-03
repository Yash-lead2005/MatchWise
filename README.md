# MatchWise – Intelligent Product Matching Platform

MatchWise is a machine learning-powered product matching and duplicate detection platform that identifies whether two product records from different e-commerce catalogues represent the same real-world product.

It combines text similarity, feature engineering, XGBoost classification, calibrated decision thresholds and human-in-the-loop review through a React dashboard.

## Features

- Product comparison using name, description and price.
- 15 engineered matching features.
- Character-level TF-IDF and cosine similarity.
- XGBoost-based match probability prediction.
- Three-level decision system:
  - AUTO_MATCH
  - AUTO_NON_MATCH
  - HUMAN_REVIEW
- Human review queue with approve/reject decisions.
- Comparison history stored in SQLite.
- Analytics dashboard with model and decision statistics.
- FastAPI backend and React frontend.

## Technology Stack

| Component | Technology |
|---|---|
| Frontend | React, TypeScript, Vite |
| Backend | Python, FastAPI |
| Machine Learning | Scikit-learn, XGBoost |
| NLP / Text Features | TF-IDF, RapidFuzz |
| Data Processing | Pandas, NumPy |
| Database | SQLite |
| Visualization | Recharts |
| Model Storage | Joblib |

## Dataset

MatchWise uses the Abt-Buy Entity Resolution Benchmark.

Source: https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution

The dataset contains product records from two catalogues and known matching pairs.

## Project Structure

```text
MatchWise/
│
├── backend/
│   ├── api/
│   │   ├── main.py
│   │   ├── database.py
│   │   ├── init_database.py
│   │   └── feature_builder.py
│   │
│   ├── ml/
│   │   ├── blocking/
│   │   ├── evaluation/
│   │   ├── features/
│   │   ├── pairing/
│   │   ├── preprocessing/
│   │   └── training/
│   │
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   ├── package.json
│   └── README.md
│
├── data/
│   ├── processed/
│   └── reports/
│
├── models/
│   ├── xgboost.joblib
│   ├── blocking_vectorizer.joblib
│   ├── text_vectorizers.joblib
│   └── xgboost_schema.json
│
├── .gitignore
└── README.md
```

## Installation and Setup

### Prerequisites

Install the following:

- Python 3.10 or later
- Node.js and npm
- Git
- VS Code (recommended)

### 1. Clone the repository

```bash
git clone https://github.com/Yash-lead2005/MatchWise.git
cd MatchWise
```

### 2. Set up the Python backend

Create a virtual environment:

```bash
python -m venv backend/venv
```

Activate it on Windows PowerShell:

```powershell
.\backend\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r backend/requirements.txt
```

### 3. Initialize the database

From the project root, with the virtual environment activated:

```bash
python backend/api/init_database.py
```

### 4. Start the backend

```bash
cd backend
uvicorn api.main:app --reload
```

Backend URL:

http://127.0.0.1:8000

Interactive API documentation:

http://127.0.0.1:8000/docs

### 5. Set up the frontend

Open a second terminal:

```bash
cd frontend
npm install
```

Start the frontend:

```bash
npm run dev
```

Frontend URL:

http://localhost:5173

## Running MatchWise Again

Whenever you reopen the project:

**Terminal 1 – Backend**

```powershell
cd "D:\AI,ML PROJECT\MatchWise"
.\backend\venv\Scripts\Activate.ps1
cd backend
uvicorn api.main:app --reload
```

**Terminal 2 – Frontend**

```powershell
cd "D:\AI,ML PROJECT\MatchWise\frontend"
npm run dev
```

Keep both terminals running while using the application.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Backend and model health |
| GET | `/model-info` | Model information |
| POST | `/predict` | Model prediction |
| POST | `/match` | Compare two products |
| GET | `/reviews/pending` | Retrieve pending reviews |
| POST | `/reviews/{record_id}/decision` | Submit human decision |
| GET | `/history` | Retrieve comparison history |
| GET | `/analytics` | Retrieve dashboard analytics |

## Machine Learning Workflow

1. Load and preprocess the Abt-Buy dataset.
2. Generate candidate product pairs using character TF-IDF and cosine similarity.
3. Extract text, price and blocking similarity features.
4. Train and evaluate multiple machine learning classifiers.
5. Select and calibrate the XGBoost model.
6. Generate match probabilities.
7. Apply calibrated decision thresholds.
8. Send uncertain comparisons for human review.
9. Store predictions and human decisions in SQLite.

## Decision System

MatchWise uses three possible outcomes:

- **AUTO_MATCH:** High-confidence product match.
- **AUTO_NON_MATCH:** High-confidence non-match.
- **HUMAN_REVIEW:** Requires manual verification.

Thresholds are calibrated using validation data.

## Important Notes

- The trained model and vectorizer files must be present inside `models/`.
- The processed datasets and reports are included in the repository.
- The local SQLite database is generated during initialization and is not committed.
- Python dependencies are installed from `backend/requirements.txt`.
- Frontend dependencies are installed using `npm install`.
- If the virtual environment is missing, recreate it using the setup instructions above.

## Author

**Yash Kumar**

GitHub: https://github.com/Yash-lead2005

---

*MatchWise – Intelligent Product Duplicate Detection and Resolution Platform.*
