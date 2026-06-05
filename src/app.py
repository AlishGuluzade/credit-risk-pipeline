import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import sqlite3
from datetime import datetime

st.set_page_config(
    page_title="Credit Risk Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "https://credit-risk-pipeline.onrender.com"

st.markdown("""
<style>
[data-testid="stSidebar"] {background-color: #1E3A5F;}
[data-testid="stSidebar"] * {color: #FFFFFF !important;}
[data-testid="stSidebar"] .stRadio label {
    font-size: 15px !important;
    padding: 8px 0px !important;
}
.metric-card {
    background: #FFFFFF;
    border: 1px solid #E8ECF0;
    border-left: 4px solid #1E3A5F;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 4px 0;
}
.metric-card h3 {color: #6B7280; font-size: 13px; margin: 0 0 4px 0; font-weight: 500;}
.metric-card h2 {color: #1A1A2E; font-size: 28px; margin: 0; font-weight: 700;}
.good-badge {
    background: #D1FAE5; color: #065F46;
    padding: 6px 16px; border-radius: 20px;
    font-weight: 600; font-size: 14px;
}
.bad-badge {
    background: #FEE2E2; color: #991B1B;
    padding: 6px 16px; border-radius: 20px;
    font-weight: 600; font-size: 14px;
}
.section-title {
    font-size: 18px; font-weight: 600;
    color: #1A1A2E; margin: 24px 0 16px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid #E8ECF0;
}
div[data-testid="stButton"] button {
    background-color: #1E3A5F !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 10px 28px !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("data/clean_credit_data.csv")

def load_predictions():
    try:
        conn = sqlite3.connect("data/credit_risk.db")
        df = pd.read_sql("SELECT * FROM predictions ORDER BY id DESC LIMIT 50", conn)
        conn.close()
        return df
    except:
        return pd.DataFrame()

def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200
    except:
        return False

df = load_data()
api_online = check_api()

# --- Sidebar ---
with st.sidebar:
    st.markdown("## Credit Risk Intelligence")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["Overview", "Analytics", "Predict", "History"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    status_color = "green" if api_online else "red"
    status_text = "Online" if api_online else "Offline"
    st.markdown(f"**API Status:** :{status_color}[{status_text}]")
    st.markdown(f"**Model:** XGBoost (tuned)")
    st.markdown(f"**AUC:** 0.7861")
    st.markdown("---")
    st.markdown("<small>Credit Risk ML Pipeline v1.0</small>", unsafe_allow_html=True)

# =================== PAGE 1: OVERVIEW ===================
if page == "Overview":
    st.markdown("# Credit Risk Intelligence")
    st.markdown("ML-powered credit risk assessment platform · XGBoost + FastAPI + SQLite")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""<div class="metric-card">
            <h3>Total Customers</h3><h2>1,000</h2></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="metric-card">
            <h3>Good Credit</h3><h2 style="color:#065F46">700</h2></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="metric-card">
            <h3>Bad Credit</h3><h2 style="color:#991B1B">300</h2></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class="metric-card">
            <h3>Model AUC</h3><h2>0.7861</h2></div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-title">Pipeline Architecture</p>', unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    for col, step, icon in zip(
        [col1, col2, col3, col4, col5],
        ["Raw Data", "ETL Pipeline", "XGBoost Model", "FastAPI", "Dashboard"],
        ["CSV", "Pandas+SQL", "MLflow+Optuna", "REST API", "Streamlit"]
    ):
        with col:
            st.markdown(f"""
            <div style="text-align:center; background:#FFFFFF;
                border:1px solid #E8ECF0; border-radius:8px; padding:16px;">
                <div style="font-weight:600; color:#1E3A5F; font-size:14px;">{step}</div>
                <div style="color:#6B7280; font-size:12px; margin-top:4px;">{icon}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-title">Model Performance Summary</p>', unsafe_allow_html=True)
    perf_data = {
        'Model': ['XGBoost (tuned)', 'LightGBM', 'Random Forest', 'Gradient Boosting'],
        'Test AUC': [0.7861, 0.7790, 0.7848, 0.7864],
        'CV AUC': [0.7859, 0.7750, 0.7810, 0.7820],
        'Accuracy': [0.785, 0.760, 0.775, 0.775],
    }
    perf_df = pd.DataFrame(perf_data)
    st.dataframe(perf_df.style.highlight_max(
        subset=['Test AUC'], color='#D1FAE5'
    ), use_container_width=True, hide_index=True)

# =================== PAGE 2: ANALYTICS ===================
elif page == "Analytics":
    st.markdown("# Analytics")
    st.markdown("Exploratory analysis of the credit risk dataset")
    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<p class="section-title">Risk Distribution</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#F8F9FA')
        ax.set_facecolor('#FFFFFF')
        counts = df['risk'].value_counts()
        wedges, texts, autotexts = ax.pie(
            counts, labels=["Good Credit", "Bad Credit"],
            autopct="%1.1f%%", colors=["#1E3A5F", "#E74C3C"],
            startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2}
        )
        for text in autotexts:
            text.set_color('white')
            text.set_fontweight('bold')
        st.pyplot(fig)
        plt.close()

    with c2:
        st.markdown('<p class="section-title">Credit Amount by Risk</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#F8F9FA')
        ax.set_facecolor('#FFFFFF')
        sns.boxplot(data=df, x="risk", y="credit_amount",
                    palette=["#1E3A5F", "#E74C3C"], ax=ax)
        ax.set_xticklabels(["Good Credit", "Bad Credit"])
        ax.set_xlabel("")
        ax.set_ylabel("Credit Amount")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<p class="section-title">Age Distribution by Risk</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#F8F9FA')
        ax.set_facecolor('#FFFFFF')
        for risk_val, color, label in [(0, "#1E3A5F", "Good"), (1, "#E74C3C", "Bad")]:
            ax.hist(df[df['risk']==risk_val]['age'], alpha=0.7,
                    color=color, label=label, bins=20, edgecolor='white')
        ax.legend()
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with c4:
        st.markdown('<p class="section-title">Feature Importance</p>', unsafe_allow_html=True)
        try:
            fi = pd.read_csv("data/feature_importance.csv").head(8)
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor('#F8F9FA')
            ax.set_facecolor('#FFFFFF')
            colors = ['#1E3A5F' if i == 0 else '#BDC3C7' for i in range(len(fi))]
            ax.barh(fi['feature'], fi['importance'], color=colors)
            ax.set_xlabel("Importance")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)
            plt.close()
        except:
            st.info("Feature importance data not found.")

    st.markdown('<p class="section-title">Duration vs Credit Amount</p>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#F8F9FA')
    ax.set_facecolor('#FFFFFF')
    for risk_val, color, label in [(0, "#1E3A5F", "Good"), (1, "#E74C3C", "Bad")]:
        subset = df[df['risk'] == risk_val]
        ax.scatter(subset['duration_months'], subset['credit_amount'],
                   alpha=0.4, color=color, label=label, s=20)
    ax.set_xlabel("Duration (months)")
    ax.set_ylabel("Credit Amount")
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)
    plt.close()

# =================== PAGE 3: PREDICT ===================
elif page == "Predict":
    st.markdown("# Risk Prediction")
    st.markdown("Enter customer details to get an instant credit risk assessment")
    st.markdown("---")

    if not api_online:
        st.warning("API is warming up... Please wait 30 seconds and refresh the page.")

    with st.form("prediction_form"):
        st.markdown('<p class="section-title">Customer Information</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.slider("Age", 18, 75, 35)
            duration = st.slider("Loan Duration (months)", 6, 72, 24)
            credit_amount = st.number_input("Credit Amount (DM)", 500, 20000, 3000, step=500)

        with c2:
            checking = st.selectbox("Checking Account Status", [0, 1, 2, 3],
                format_func=lambda x: ["Negative balance", "Low (< 200 DM)",
                                        "Medium (≥ 200 DM)", "No account"][x])
            savings = st.selectbox("Savings Account", [0, 1, 2, 3, 4],
                format_func=lambda x: ["Unknown/None", "Low (< 100 DM)", "Medium",
                                        "High (≥ 500 DM)", "Very High (≥ 1000 DM)"][x])
            employment = st.selectbox("Employment Duration", [0, 1, 2, 3, 4],
                format_func=lambda x: ["Unemployed", "< 1 year", "1–4 years",
                                        "4–7 years", "> 7 years"][x])

        with c3:
            housing = st.selectbox("Housing", [0, 1, 2],
                format_func=lambda x: ["Free", "Own", "Rent"][x])
            job = st.selectbox("Job Level", [0, 1, 2, 3],
                format_func=lambda x: ["Unskilled (non-resident)", "Unskilled (resident)",
                                        "Skilled", "Highly Skilled"][x])
            installment = st.slider("Installment Rate (% of income)", 1, 4, 2)

        submitted = st.form_submit_button("Assess Credit Risk")

    if submitted:
        if not api_online:
            st.error("Cannot predict — API is offline.")
        else:
            payload = {
                "age": age, "duration_months": duration,
                "credit_amount": float(credit_amount),
                "checking_account": checking, "savings_account": savings,
                "employment_years": employment, "housing": housing,
                "job": job, "installment_rate": installment
            }
            try:
                response = requests.post(f"{API_URL}/predict", json=payload)
                result = response.json()
                st.markdown("---")
                st.markdown('<p class="section-title">Assessment Result</p>', unsafe_allow_html=True)

                rc1, rc2, rc3, rc4 = st.columns(4)
                badge = "good-badge" if result['prediction'] == "Good Risk" else "bad-badge"
                with rc1:
                    st.markdown(f"""<div class="metric-card">
                        <h3>Decision</h3>
                        <div class="{badge}" style="margin-top:8px">{result['prediction']}</div>
                    </div>""", unsafe_allow_html=True)
                with rc2:
                    st.markdown(f"""<div class="metric-card">
                        <h3>Risk Probability</h3>
                        <h2>{result['risk_probability']*100:.1f}%</h2>
                    </div>""", unsafe_allow_html=True)
                with rc3:
                    st.markdown(f"""<div class="metric-card">
                        <h3>Confidence</h3>
                        <h2>{result['confidence']*100:.1f}%</h2>
                    </div>""", unsafe_allow_html=True)
                with rc4:
                    st.markdown(f"""<div class="metric-card">
                        <h3>Monthly Payment</h3>
                        <h2>{credit_amount/duration:.0f} DM</h2>
                    </div>""", unsafe_allow_html=True)

                prob = result['risk_probability']
                st.markdown('<p class="section-title">Risk Gauge</p>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(8, 1.5))
                fig.patch.set_facecolor('#F8F9FA')
                ax.set_facecolor('#F8F9FA')
                ax.barh(0, 1, color='#E8ECF0', height=0.4)
                color = "#E74C3C" if prob > 0.5 else "#1E3A5F"
                ax.barh(0, prob, color=color, height=0.4)
                ax.axvline(x=0.5, color='#6B7280', linestyle='--', linewidth=1)
                ax.set_xlim(0, 1)
                ax.set_yticks([])
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
                ax.set_xticklabels(['0%', '25%', '50% (threshold)', '75%', '100%'])
                ax.spines[:].set_visible(False)
                st.pyplot(fig)
                plt.close()

            except Exception as e:
                st.error(f"API Error: {e}")

# =================== PAGE 4: HISTORY ===================
elif page == "History":
    st.markdown("# Prediction History")
    st.markdown("All predictions saved to SQLite database via FastAPI")
    st.markdown("---")

    preds = load_predictions()
    if not preds.empty:
        total = len(preds)
        good = len(preds[preds['prediction'] == 0])
        bad = len(preds[preds['prediction'] == 1])
        avg_prob = preds['probability'].mean()

        h1, h2, h3, h4 = st.columns(4)
        with h1:
            st.markdown(f"""<div class="metric-card">
                <h3>Total Predictions</h3><h2>{total}</h2></div>""", unsafe_allow_html=True)
        with h2:
            st.markdown(f"""<div class="metric-card">
                <h3>Good Risk</h3><h2 style="color:#065F46">{good}</h2></div>""", unsafe_allow_html=True)
        with h3:
            st.markdown(f"""<div class="metric-card">
                <h3>Bad Risk</h3><h2 style="color:#991B1B">{bad}</h2></div>""", unsafe_allow_html=True)
        with h4:
            st.markdown(f"""<div class="metric-card">
                <h3>Avg Risk Prob</h3><h2>{avg_prob*100:.1f}%</h2></div>""", unsafe_allow_html=True)

        st.markdown('<p class="section-title">Recent Predictions</p>', unsafe_allow_html=True)
        preds_display = preds.copy()
        preds_display['prediction'] = preds_display['prediction'].map({0: "Good Risk", 1: "Bad Risk"})
        preds_display['probability'] = (preds_display['probability'] * 100).round(1).astype(str) + "%"
        preds_display['predicted_at'] = pd.to_datetime(
            preds_display['predicted_at']).dt.strftime('%Y-%m-%d %H:%M')
        st.dataframe(preds_display, use_container_width=True, hide_index=True)
    else:
        st.info("No predictions yet. Go to the Predict page to make your first assessment.")