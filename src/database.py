import sqlite3
import pandas as pd
import os
from datetime import datetime

DB_PATH = "data/credit_risk.db"

def create_connection():
    conn = sqlite3.connect(DB_PATH)
    return conn

def create_tables():
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            age INTEGER,
            duration_months INTEGER,
            credit_amount REAL,
            checking_account INTEGER,
            savings_account INTEGER,
            employment_years INTEGER,
            housing INTEGER,
            job INTEGER,
            installment_rate INTEGER,
            monthly_payment REAL,
            high_credit INTEGER,
            risk INTEGER,
            created_at TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER,
            model_name TEXT,
            prediction INTEGER,
            probability REAL,
            predicted_at TEXT,
            FOREIGN KEY (customer_id) REFERENCES customers(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT,
            test_auc REAL,
            cv_auc REAL,
            accuracy REAL,
            precision REAL,
            recall REAL,
            logged_at TEXT
        )
    """)

    conn.commit()
    conn.close()
    print("[OK] Tables created: customers, predictions, model_metrics")

def load_csv_to_db():
    conn = create_connection()
    df = pd.read_csv("data/clean_credit_data.csv")
    df['created_at'] = datetime.now().isoformat()

    cols = ['age', 'duration_months', 'credit_amount', 'checking_account',
            'savings_account', 'employment_years', 'housing', 'job',
            'installment_rate', 'monthly_payment', 'high_credit', 'risk', 'created_at']

    df[cols].to_sql('customers', conn, if_exists='replace', index=False)
    conn.close()
    print(f"[OK] {len(df)} records loaded into customers table")

def log_model_metrics(model_name, test_auc, cv_auc, accuracy, precision, recall):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO model_metrics
        (model_name, test_auc, cv_auc, accuracy, precision, recall, logged_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (model_name, test_auc, cv_auc, accuracy, precision, recall,
          datetime.now().isoformat()))
    conn.commit()
    conn.close()
    print(f"[OK] Metrics logged for: {model_name}")

def log_prediction(customer_data, prediction, probability, model_name="XGBoost"):
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO customers
        (age, duration_months, credit_amount, checking_account, savings_account,
         employment_years, housing, job, installment_rate, monthly_payment,
         high_credit, risk, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (*customer_data, datetime.now().isoformat()))

    customer_id = cursor.lastrowid

    cursor.execute("""
        INSERT INTO predictions
        (customer_id, model_name, prediction, probability, predicted_at)
        VALUES (?, ?, ?, ?, ?)
    """, (customer_id, model_name, prediction, probability,
          datetime.now().isoformat()))

    conn.commit()
    conn.close()
    return customer_id

def get_stats():
    conn = create_connection()
    print("\n--- Database Statistics ---")
    print(pd.read_sql("SELECT COUNT(*) as total, risk FROM customers GROUP BY risk", conn).to_string(index=False))
    print("\nLatest predictions:")
    print(pd.read_sql("SELECT * FROM predictions ORDER BY id DESC LIMIT 5", conn).to_string(index=False))
    print("\nModel metrics:")
    print(pd.read_sql("SELECT * FROM model_metrics", conn).to_string(index=False))
    conn.close()

def run_database_pipeline():
    print("=" * 40)
    print("Database Pipeline Started")
    print("=" * 40)
    os.makedirs("data", exist_ok=True)
    create_tables()
    load_csv_to_db()

    log_model_metrics(
        model_name="XGBoost (tuned)",
        test_auc=0.7861,
        cv_auc=0.7859,
        accuracy=0.785,
        precision=0.7297,
        recall=0.45
    )

    get_stats()
    print("\n[DONE] Database pipeline completed!")

if __name__ == "__main__":
    run_database_pipeline()