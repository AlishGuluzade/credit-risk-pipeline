import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import optuna
import pickle
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

optuna.logging.set_verbosity(optuna.logging.WARNING)

MLFLOW_EXPERIMENT = "credit-risk-models"

def load_clean_data(filepath):
    df = pd.read_csv(filepath)
    print(f"[OK] Clean data loaded: {df.shape}")
    return df

def prepare_features(df):
    X = df.drop(columns=['risk'])
    y = df['risk']
    print(f"[OK] Features: {X.shape[1]} columns | Target: {y.value_counts().to_dict()}")
    return X, y

def cross_validate_model(model, X, y, name):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    print(f"  {name} CV AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")
    return scores.mean()

def optimize_xgboost(X_train, y_train):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'random_state': 42
        }
        model = XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        score = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc').mean()
        return score

    print("[*] Optimizing XGBoost with Optuna (30 trials)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    print(f"[OK] Best XGBoost params found | AUC: {study.best_value:.4f}")
    return study.best_params

def train_all_models(X_train, y_train, X, y, best_xgb_params):
    print("\n--- Training & Cross-Validating All Models ---")

    models = {
        'XGBoost (tuned)': XGBClassifier(
            **best_xgb_params,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            random_state=42,
            verbose=-1
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200,
            random_state=42
        ),
    }

    cv_scores = {}
    for name, model in models.items():
        score = cross_validate_model(model, X, y, name)
        cv_scores[name] = score

    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model

    return trained, cv_scores

def evaluate_and_log(models, cv_scores, X_test, y_test):
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    results = {}

    print("\n--- Model Evaluation + MLflow Logging ---")
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
            report = classification_report(y_test, y_pred, output_dict=True)

            results[name] = {
                'auc': round(auc, 4),
                'cv_auc': round(cv_scores[name], 4),
                'accuracy': round(report['accuracy'], 4),
                'precision': round(report['1']['precision'], 4),
                'recall': round(report['1']['recall'], 4),
            }

            mlflow.log_metric("test_auc", auc)
            mlflow.log_metric("cv_auc", cv_scores[name])
            mlflow.log_metric("accuracy", report['accuracy'])
            mlflow.log_metric("precision", report['1']['precision'])
            mlflow.log_metric("recall", report['1']['recall'])
            mlflow.sklearn.log_model(model, "model")

            print(f"\n{name}:")
            print(f"  Test AUC:  {auc:.4f} | CV AUC: {cv_scores[name]:.4f}")
            print(f"  Accuracy:  {report['accuracy']:.4f}")
            print(f"  Precision: {report['1']['precision']:.4f}")
            print(f"  Recall:    {report['1']['recall']:.4f}")

    return results

def save_best_model(models, results):
    best_name = max(results, key=lambda x: results[x]['auc'])
    best_model = models[best_name]
    os.makedirs('models', exist_ok=True)
    with open('models/best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print(f"\n[OK] Best model: {best_name}")
    print(f"     Test AUC: {results[best_name]['auc']} | CV AUC: {results[best_name]['cv_auc']}")
    print("[OK] Saved: models/best_model.pkl")
    return best_name

def get_feature_importance(models, X):
    for name in ['XGBoost (tuned)', 'LightGBM', 'Random Forest']:
        if name in models:
            model = models[name]
            importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            importance.to_csv('data/feature_importance.csv', index=False)
            print(f"\n--- Top 5 Features ({name}) ---")
            print(importance.head(5).to_string(index=False))
            break

def run_model(input_path):
    print("=" * 40)
    print("ML Pipeline Started")
    print("=" * 40)

    df = load_clean_data(input_path)
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[OK] Train: {X_train.shape} | Test: {X_test.shape}")

    best_xgb_params = optimize_xgboost(X_train, y_train)
    models, cv_scores = train_all_models(X_train, y_train, X, y, best_xgb_params)
    results = evaluate_and_log(models, cv_scores, X_test, y_test)
    get_feature_importance(models, X)
    save_best_model(models, results)

    print("\n[DONE] ML pipeline completed successfully!")
    return models, results

if __name__ == "__main__":
    run_model("data/clean_credit_data.csv")