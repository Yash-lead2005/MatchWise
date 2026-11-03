# MatchWise – ML-Powered Product Matching System

MatchWise is a machine-learning-based product matching system that identifies whether two product records represent the same real-world product.

It uses text similarity, feature engineering, and an XGBoost classifier to predict product matches. It also supports confidence-based decisions, human review, prediction history, and analytics through a web dashboard.

## Key Features

- **Product comparison:** Compare two product records using their names, descriptions, and prices.
- **Feature engineering:** Generate 15 comparison features, including character similarity, token similarity, Jaccard similarity, TF-IDF similarity, price differences, and blocking similarity.
- **Machine learning:** Use a trained XGBoost classification pipeline to estimate match probability.
- **Three-level decision system:**
  - `AUTO_MATCH` – sufficiently high match confidence.
  - `AUTO_NON_MATCH` – sufficiently low match confidence.
  - `HUMAN_REVIEW` – cases requiring manual verification.
- **Human review:** Approve or reject pending product matches.
- **Matching history:** Store and view previous comparisons and final decisions.
- **Analytics dashboard:** View matching and review statistics.
- **REST API:** Access the backend through FastAPI and its Swagger documentation.

## Technology Stack

| Component | Technologies |
|---|---|
| Frontend | React, TypeScript, Vite |
| UI and charts | CSS, Lucide React, Recharts |
| Backend | Python, FastAPI |
| Machine learning | XGBoost, Scikit-learn |
| Data processing | Pandas, NumPy |
| Text similarity | TF-IDF, RapidFuzz |
| Database | SQLite |
| Model persistence | Joblib |

## Project Structure

```text
MatchWise/
│
├── backend/
│   ├── api/
│   │   ├── database.py
│   │   ├── feature_builder.py
│   │   ├── init_database.py
│   │   └── main.py
│   │
│   ├── ml/
│   │   ├── blocking/
│   │   ├── evaluation/
│   │   ├── features/
│   │   ├── pairing/
│   │   ├── preprocessing/
│   │   └── training/
│   │
│   ├── requirements.txt
│   └── venv/                 # Created locally; not committed
│
├── frontend/
│   ├── public/
│   ├── src/
│   ├── package.json
│   └── package-lock.json
│
├── models/
│   ├── xgboost.joblib
│   ├── xgboost_schema.json
│   ├── text_vectorizers.joblib
│   └── blocking_vectorizer.joblib
│
├── data/
│   ├── processed/
│   ├── reports/
│   └── database/
│
├── README.md
└── .gitignore
```

*Some generated data files and local environment folders are intentionally excluded from version control.*

## Getting Started – Windows

Follow these steps to run MatchWise on a new Windows machine.

### 1. Install prerequisites

Install:

- Python 3.10 or a compatible version supported by the project's dependencies.
- Node.js and npm.
- Git.
- Visual Studio Code (recommended).

Verify the installations:

```powershell
python --version
node --version
npm --version
git --version
```

### 2. Clone the GitHub repository

```powershell
git clone YOUR_GITHUB_REPOSITORY_URL
cd MatchWise
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with the repository URL.

### 3. Set up the backend

Open a terminal in VS Code:

```powershell
cd backend

python -m venv venv

.\venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

pip install -r requirements.txt
```

If PowerShell blocks environment activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### 4. Start the backend

Make sure the terminal is inside the `backend` folder and the virtual environment is activated.

```powershell
uvicorn api.main:app --reload
```

Backend address:

```text
http://127.0.0.1:8000
```

Swagger API documentation:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/health
```

### 5. Set up the frontend

Open a **second terminal**. Keep the backend running.

```powershell
cd frontend

npm install

npm run dev
```

Open the frontend in your browser:

```text
http://localhost:5173
```

## Running MatchWise Again

Whenever you want to run the project in the future, follow these two steps.

**Terminal 1 – Backend**

```powershell
cd "D:\AI,ML PROJECT\MatchWise\backend"

.\venv\Scripts\Activate.ps1

uvicorn api.main:app --reload
```

**Terminal 2 – Frontend**

```powershell
cd "D:\AI,ML PROJECT\MatchWise\frontend"

npm run dev
```

Then open:

```text
http://localhost:5173
```

Keep both terminals running while using the application.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check backend and model status |
| GET | `/model-info` | View model information |
| POST | `/predict` | Generate a model prediction |
| POST | `/match` | Compare products and save the result |
| GET | `/reviews/pending` | Retrieve pending human reviews |
| POST | `/reviews/{record_id}/decision` | Submit a human decision |
| GET | `/history` | Retrieve comparison history |
| GET | `/analytics` | Retrieve dashboard statistics |

## Machine Learning Workflow

1. Load and preprocess product catalogues.
2. Generate candidate product pairs using blocking.
3. Engineer comparison features for candidate pairs.
4. Train and evaluate the XGBoost classifier.
5. Calibrate decision thresholds using validation data.
6. Save the trained model, schema, and text vectorizers.
7. Generate predictions through the FastAPI backend.
8. Store predictions and human decisions in SQLite.
9. Display results, history, and analytics in React.

## Model Artifacts

The backend requires the trained model and supporting artifacts in the root `models/` directory.

Important files:

- `xgboost.joblib` – trained XGBoost pipeline.
- `xgboost_schema.json` – feature column order and model threshold.
- `text_vectorizers.joblib` – saved text vectorizers.
- `blocking_vectorizer.joblib` – saved candidate-blocking vectorizer.

If these files are not included in the repository, they must be generated or provided separately before running the application.

## Notes

- The application is configured for local development.
- The SQLite database stores comparison and human-review records locally.
- Training datasets and generated files may need to be obtained separately if excluded from GitHub.
- Keep local virtual environments, dependency folders, database files, and secrets out of version control.
- Model predictions are estimates and should be reviewed when the system returns `HUMAN_REVIEW`.

## Author

**Yash Kumar**

Machine Learning | Python | FastAPI | React | XGBoost

---

**MatchWise – Intelligent Product Matching Through Machine Learning.**