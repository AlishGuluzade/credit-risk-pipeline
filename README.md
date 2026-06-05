# Credit Risk Intelligence Pipeline

> An end-to-end machine learning system that predicts whether a bank customer will repay their loan or default — built with XGBoost, FastAPI, MLflow, Streamlit, and SQLite.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-AUC%200.786-brightgreen?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?style=flat-square)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?style=flat-square)

---

## What is this project?

Imagine a bank that gives out loans every day. Some customers repay their loans, others don't. The bank wants to know **in advance** whether a new customer is likely to default — before approving the loan.

This project builds a complete, production-style ML system that does exactly that. You enter a customer's details (age, income, loan amount, employment status, etc.), and the system instantly predicts: **Good Risk** or **Bad Risk** — along with a probability score and confidence level.

The system is not just a model. It's a full pipeline:
- Raw data is ingested and cleaned automatically
- Multiple ML models are trained and compared
- The best model is served via a REST API
- Every prediction is logged to a database
- Everything is visualized in a multi-page dashboard

---

## System Architecture

```
Raw CSV Data
     │
     ▼
┌─────────────────────┐
│   ETL Pipeline      │  ← Pandas, NumPy
│   (etl.py)          │    Clean, rename, feature engineer
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│   SQLite Database   │  ← 3 tables: customers, predictions, model_metrics
│   (database.py)     │    Real database instead of CSV files
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│   ML Pipeline       │  ← XGBoost, LightGBM, Random Forest, Gradient Boosting
│   (model.py)        │    Optuna hyperparameter tuning (30 trials)
│                     │    5-fold Cross-Validation
│                     │    MLflow experiment tracking
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│   FastAPI           │  ← REST API with Swagger UI documentation
│   (api.py)          │    POST /predict → returns risk prediction
│                     │    GET /stats → returns database statistics
│                     │    Every prediction auto-saved to SQLite
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│   Streamlit         │  ← Multi-page dashboard with custom theme
│   (app.py)          │    Overview / Analytics / Predict / History
│                     │    Calls FastAPI in real-time
└─────────────────────┘
```

---

## Tech Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| Language | Python 3.10 | Core language |
| Data Processing | Pandas, NumPy | ETL, cleaning, feature engineering |
| Machine Learning | XGBoost, LightGBM, Scikit-learn | Model training and evaluation |
| Hyperparameter Tuning | Optuna | Automated optimization (30 trials) |
| Cross-Validation | Scikit-learn StratifiedKFold | 5-fold CV for reliable evaluation |
| Experiment Tracking | MLflow | Logs every model run with metrics |
| Database | SQLite + SQLAlchemy | Stores customers, predictions, metrics |
| API | FastAPI + Uvicorn | Serves the model as a REST endpoint |
| API Documentation | Swagger UI (auto-generated) | Interactive API testing |
| Dashboard | Streamlit | Multi-page frontend |
| Version Control | Git + GitHub | Source control |

---

## Machine Learning Results

Four models were trained and evaluated. XGBoost was selected as the final model because it achieved the best balance between Test AUC and CV AUC — meaning it generalizes well to unseen data without overfitting.

| Model | Test AUC | CV AUC | Accuracy | Precision | Recall |
|-------|----------|--------|----------|-----------|--------|
| **XGBoost (tuned)** | **0.7861** | **0.7859** | **0.785** | **0.7297** | **0.450** |
| Gradient Boosting | 0.7864 | 0.7820 | 0.775 | 0.6829 | 0.467 |
| Random Forest | 0.7848 | 0.7810 | 0.775 | 0.7778 | 0.350 |
| LightGBM | 0.7790 | 0.7750 | 0.760 | 0.6800 | 0.420 |

**Why XGBoost?**
The gap between XGBoost's Test AUC (0.7861) and CV AUC (0.7859) is only 0.0002 — the smallest among all models. This means the model is not memorizing training data; it's learning real patterns that work on new customers.

**What does AUC mean?**
- 0.5 = random guessing (flipping a coin)
- 0.786 = correctly identifies risk 78.6% of the time
- 1.0 = perfect prediction

---

## Top Predictive Features

These are the features the XGBoost model found most useful when making decisions:

| Rank | Feature | Importance | What it means |
|------|---------|-----------|---------------|
| 1 | monthly_payment | 0.1172 | Loan amount ÷ duration (engineered feature) |
| 2 | credit_amount | 0.1145 | Total loan amount requested |
| 3 | checking_account | 0.0956 | Status of the customer's checking account |
| 4 | age | 0.0851 | Customer's age |
| 5 | duration_months | 0.0823 | Loan repayment period in months |

Note: `monthly_payment` is a **new feature we created** during feature engineering — it wasn't in the original dataset. It turned out to be the single most important predictor.

---

## Dataset

- **Name:** German Credit Dataset
- **Source:** UCI Machine Learning Repository (Statlog)
- **Records:** 1,000 bank customers
- **Original Features:** 21 (all in German, all numerically encoded)
- **Engineered Features:** 3 new features added
- **Target Variable:** Credit risk (1 = Good, 0 = Bad)
- **Class Distribution:** 700 Good (70%), 300 Bad (30%)

### Feature Engineering

Three new features were created from existing data:

1. **monthly_payment** = `credit_amount ÷ duration_months`
   Captures how large each monthly payment is relative to the loan size. High monthly payments increase default risk.

2. **high_credit** = 1 if `credit_amount > median`, else 0
   Binary flag for customers requesting above-average loan amounts.

3. **age_group** = Bucketed age into 4 groups
   young (≤25), adult (26–35), middle_aged (36–50), senior (51+)

---

## Project Structure

```
credit-risk-pipeline/
│
├── data/
│   └── german_credit_data.csv       ← Raw dataset (original, uncleaned)
│
├── src/
│   ├── etl.py                       ← ETL pipeline: clean, rename, engineer features
│   ├── model.py                     ← Train 4 models, Optuna tuning, MLflow logging
│   ├── database.py                  ← SQLite setup, data loading, prediction logging
│   ├── api.py                       ← FastAPI REST API with /predict and /stats endpoints
│   └── app.py                       ← Streamlit multi-page dashboard
│
├── models/
│   └── best_model.pkl               ← Saved XGBoost model (generated after training)
│
├── .streamlit/
│   └── config.toml                  ← Custom Streamlit theme (navy blue + white)
│
├── requirements.txt                 ← All Python dependencies
└── README.md                        ← This file
```

---

## Dashboard Pages

The Streamlit dashboard has 4 pages accessible from the sidebar:

### 1. Overview
- KPI cards: total customers, good credit count, bad credit count, model AUC
- Pipeline architecture visualization (Raw Data → ETL → Model → API → Dashboard)
- Model performance comparison table with best model highlighted

### 2. Analytics
- Risk distribution pie chart (70% Good / 30% Bad)
- Credit amount by risk group (box plot)
- Age distribution by risk (histogram overlay)
- Feature importance bar chart
- Duration vs Credit Amount scatter plot

### 3. Predict
- Customer input form (age, loan amount, duration, employment, etc.)
- Sends data to FastAPI endpoint in real-time
- Displays: prediction label, risk probability, confidence score, monthly payment
- Visual risk gauge bar showing probability vs 50% threshold
- Prediction automatically saved to SQLite database

### 4. History
- Live table of all predictions pulled from SQLite database
- Summary metrics: total predictions, good/bad counts, average risk probability

---

## API Endpoints

The FastAPI backend exposes the following endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API status check |
| GET | `/health` | Model health check |
| POST | `/predict` | Submit customer data, receive risk prediction |
| GET | `/stats` | Prediction statistics from database |

### Example Request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "duration_months": 24,
    "credit_amount": 5000,
    "checking_account": 1,
    "savings_account": 2,
    "employment_years": 3,
    "housing": 2,
    "job": 3,
    "installment_rate": 2
  }'
```

### Example Response

```json
{
  "prediction": "Good Risk",
  "risk_probability": 0.2615,
  "confidence": 0.7385,
  "model": "XGBoost (tuned)"
}
```

---

## Installation & Usage

### Prerequisites
- Python 3.10+
- Git

### Step 1: Clone the repository
```bash
git clone https://github.com/AlishGuluzade/credit-risk-pipeline.git
cd credit-risk-pipeline
```

### Step 2: Create and activate virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the ETL pipeline
```bash
python src/etl.py
```
This cleans the raw data, renames columns, engineers new features, and saves `clean_credit_data.csv`.

### Step 5: Train the models
```bash
python src/model.py
```
This trains 4 models, runs Optuna tuning, logs everything to MLflow, and saves the best model as `models/best_model.pkl`.

### Step 6: Set up the database
```bash
python src/database.py
```
This creates the SQLite database with 3 tables and loads the cleaned data.

### Step 7: Start the FastAPI server
```bash
uvicorn src.api:app --reload
```
API available at: `http://127.0.0.1:8000`
Swagger docs at: `http://127.0.0.1:8000/docs`

### Step 8: Launch the Streamlit dashboard
Open a new terminal, activate venv, then:
```bash
streamlit run src/app.py
```
Dashboard available at: `http://localhost:8501`

### Optional: View MLflow experiment tracking UI
```bash
mlflow ui
```
MLflow UI available at: `http://127.0.0.1:5000`

---

## Key Design Decisions

**Why XGBoost over Gradient Boosting?**
Gradient Boosting had a slightly higher Test AUC (0.7864 vs 0.7861), but XGBoost's CV AUC was much closer to its Test AUC. This indicates XGBoost is more stable and less likely to overfit on new production data.

**Why SQLite instead of just CSV files?**
Real production systems use databases, not flat files. SQLite allows structured queries, relational data (customers linked to predictions), and concurrent read access from both the API and dashboard simultaneously.

**Why FastAPI instead of calling the model directly from Streamlit?**
Decoupling the model from the frontend is a production best practice. The API can be consumed by any frontend (mobile app, another web service, third-party integrations) — not just Streamlit. It also means the model can be updated without touching the dashboard code.

**Why Optuna for tuning?**
Grid search checks every combination manually. Optuna uses Bayesian optimization to intelligently search the parameter space — finding better results in fewer trials.

---

## Author

**Alish Guluzade**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/alishguluzade)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github)](https://github.com/AlishGuluzade)
