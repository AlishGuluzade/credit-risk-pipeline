import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

def load_data(filepath):
    df = pd.read_csv(filepath)
    print(f"[OK] Data loaded: {df.shape}")
    return df

def rename_columns(df):
    col_map = {
        'laufkont':  'checking_account',
        'laufzeit':  'duration_months',
        'moral':     'credit_history',
        'verw':      'purpose',
        'hoehe':     'credit_amount',
        'sparkont':  'savings_account',
        'beszeit':   'employment_years',
        'rate':      'installment_rate',
        'famges':    'personal_status',
        'buerge':    'guarantors',
        'wohnzeit':  'residence_years',
        'verm':      'property',
        'alter':     'age',
        'weitkred':  'other_credits',
        'wohn':      'housing',
        'bishkred':  'existing_credits',
        'beruf':     'job',
        'pers':      'dependents',
        'telef':     'telephone',
        'gastarb':   'foreign_worker',
        'kredit':    'risk'
    }
    df = df.rename(columns=col_map)
    print("[OK] Columns renamed")
    return df

def clean_data(df):
    df['risk'] = df['risk'].apply(lambda x: 0 if x == 1 else 1)
    df['checking_account'] = df['checking_account'].map({
        1: 'negative', 2: 'low', 3: 'medium', 4: 'high'
    })
    df['savings_account'] = df['savings_account'].map({
        1: 'unknown', 2: 'low', 3: 'medium', 4: 'high', 5: 'very_high'
    })
    df['housing'] = df['housing'].map({1: 'free', 2: 'own', 3: 'rent'})
    df['employment_years'] = df['employment_years'].map({
        1: 'unemployed', 2: 'less_1yr', 3: '1_to_4yr', 4: '4_to_7yr', 5: 'over_7yr'
    })
    df['job'] = df['job'].map({
        1: 'unskilled_nonresident', 2: 'unskilled_resident',
        3: 'skilled', 4: 'highly_skilled'
    })
    print("[OK] Data cleaned")
    return df

def feature_engineering(df):
    df['age_group'] = pd.cut(
        df['age'],
        bins=[0, 25, 35, 50, 100],
        labels=['young', 'adult', 'middle_aged', 'senior']
    )
    df['monthly_payment'] = (df['credit_amount'] / df['duration_months']).round(2)
    df['high_credit'] = (df['credit_amount'] > df['credit_amount'].median()).astype(int)
    print("[OK] New features added")
    return df

def encode_features(df):
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    print(f"[OK] {len(cat_cols)} categorical columns encoded")
    return df

def save_data(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[OK] Clean data saved: {output_path}")

def run_etl(input_path, output_path):
    print("=" * 40)
    print("ETL Pipeline Started")
    print("=" * 40)
    df = load_data(input_path)
    df = rename_columns(df)
    df = clean_data(df)
    df = feature_engineering(df)

    print("\n--- Data Statistics ---")
    print(f"Risk distribution:\n{df['risk'].value_counts()}")
    print(f"\nAverage credit amount: {df['credit_amount'].mean():.0f}")
    print(f"Average duration: {df['duration_months'].mean():.1f} months")

    df = encode_features(df)
    save_data(df, output_path)
    print("\n[DONE] ETL pipeline completed successfully!")
    print(f"Final shape: {df.shape}")
    return df

if __name__ == "__main__":
    run_etl(
        input_path="data/german_credit_data.csv",
        output_path="data/clean_credit_data.csv"
    )