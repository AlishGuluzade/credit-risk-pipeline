from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
from src.database import log_prediction

app = FastAPI(
    title="Credit Risk Prediction API",
    description="ML-powered credit risk assessment using XGBoost",
    version="1.0.0"
)

with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

class CustomerInput(BaseModel):
    age: int
    duration_months: int
    credit_amount: float
    checking_account: int
    savings_account: int
    employment_years: int
    housing: int
    job: int
    installment_rate: int

class PredictionOutput(BaseModel):
    prediction: str
    risk_probability: float
    confidence: float
    model: str

@app.get("/")
def root():
    return {"message": "Credit Risk API is running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy", "model": "XGBoost (tuned)"}

@app.post("/predict", response_model=PredictionOutput)
def predict(customer: CustomerInput):
    monthly_payment = round(customer.credit_amount / customer.duration_months, 2)
    high_credit = 1 if customer.credit_amount > 2745 else 0
    age_group = 0 if customer.age <= 25 else (1 if customer.age <= 35 else (2 if customer.age <= 50 else 3))

    input_df = pd.DataFrame([{
        'checking_account': customer.checking_account,
        'duration_months': customer.duration_months,
        'credit_history': 2,
        'purpose': 2,
        'credit_amount': customer.credit_amount,
        'savings_account': customer.savings_account,
        'employment_years': customer.employment_years,
        'installment_rate': customer.installment_rate,
        'personal_status': 1,
        'guarantors': 0,
        'residence_years': 2,
        'property': 1,
        'age': customer.age,
        'other_credits': 2,
        'housing': customer.housing,
        'existing_credits': 1,
        'job': customer.job,
        'dependents': 1,
        'telephone': 0,
        'foreign_worker': 0,
        'age_group': age_group,
        'monthly_payment': monthly_payment,
        'high_credit': high_credit
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    risk_prob = float(probability[1])
    confidence = float(max(probability))

    customer_data = (
        customer.age, customer.duration_months, customer.credit_amount,
        customer.checking_account, customer.savings_account,
        customer.employment_years, customer.housing, customer.job,
        customer.installment_rate, monthly_payment, high_credit, int(prediction)
    )
    log_prediction(customer_data, int(prediction), risk_prob)

    return PredictionOutput(
        prediction="Bad Risk" if prediction == 1 else "Good Risk",
        risk_probability=round(risk_prob, 4),
        confidence=round(confidence, 4),
        model="XGBoost (tuned)"
    )

@app.get("/stats")
def stats():
    import sqlite3
    conn = sqlite3.connect("data/credit_risk.db")
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM predictions")
    total_preds = cursor.fetchone()[0]
    cursor.execute("SELECT prediction, COUNT(*) FROM predictions GROUP BY prediction")
    dist = cursor.fetchall()
    cursor.execute("SELECT * FROM model_metrics ORDER BY id DESC LIMIT 1")
    metrics = cursor.fetchone()
    conn.close()
    return {
        "total_predictions": total_preds,
        "distribution": {"good": next((d[1] for d in dist if d[0]==0), 0),
                        "bad": next((d[1] for d in dist if d[0]==1), 0)},
        "best_model_auc": metrics[2] if metrics else None
    }