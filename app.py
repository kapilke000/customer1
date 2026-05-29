import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page Configuration (Dark Premium Theme)
st.set_page_config(
    page_title="Analytics Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Glassmorphism & Cyberpunk-ish Premium Aesthetics
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: #0f172a;
        color: #f8fafc;
    }
    /* Global Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #1e293b !important;
    }
    /* Premium Action Button */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 16px;
        font-weight: 700;
        font-size: 18px;
        letter-spacing: 1px;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
        transition: all 0.4s ease;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.6);
        background: linear-gradient(135deg, #60a5fa 0%, #2563eb 100%);
    }
    /* Modern Glassmorphism Cards */
    .metric-box {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin-top: 15px;
    }
    .metric-lbl {
        font-size: 12px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }
    .metric-val {
        font-size: 32px;
        font-weight: 800;
        margin-top: 8px;
    }
    /* Custom Headers */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 800 !important;
    }
    /* Style form fields text */
    label p {
        color: #cbd5e1 !important;
        font-weight: 500 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Safe Model Loading
@st.cache_resource
def load_model():
    try:
        return joblib.load("Logistic.pkl")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #3b82f6;'>🔮 ENGINE v1.0</h2>", unsafe_allow_html=True)
    st.write("---")
    st.markdown("### 📋 Quick Summary")
    st.write("Use the interactive tabs on the right side to configure user metrics. Once configured, initiate the matrix analysis.")
    st.write("---")
    st.markdown("✨ **Status:** Connected to `Logistic.pkl`")
    st.caption("Powered by Scikit-Learn & Streamlit UI")

# --- MAIN DASHBOARD HEADER ---
st.markdown("<h1 style='font-size: 42px; margin-bottom: 5px;'>⚡ Customer Risk Intelligence Platform</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #94a3b8; font-size: 18px;'>AI-powered forecasting dashboard to track user health scores and attrition alerts.</p>", unsafe_allow_html=True)
st.write("---")

if model is not None:
    # Interactive Tabs for inputs
    tab1, tab2, tab3 = st.tabs(["👤 User Demographics", "🔌 Service Architecture", "💳 Financial Configurations"])

    with tab1:
        st.write("##")
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Select Gender Identity", ["Male", "Female"])
            senior_citizen = st.selectbox("Is Customer a Senior Citizen? (Age ≥ 65)", ["No", "Yes"])
        with col2:
            partner = st.selectbox("Relationship Status (Has Partner?)", ["No", "Yes"])
            dependents = st.selectbox("Dependent Status (Has Family Dependents?)", ["No", "Yes"])

    with tab2:
        st.write("##")
        col1, col2, col3 = st.columns(3)
        with col1:
            phone_service = st.selectbox("Primary Phone Service", ["No", "Yes"])
            multiple_lines = st.selectbox("Multiple Line Multi-plexing", ["No", "No phone service", "Yes"])
            internet_service = st.selectbox("Core Internet Framework", ["DSL", "Fiber optic", "No"])
        with col2:
            online_security = st.selectbox("Online Cyber Security Protection", ["No", "No internet service", "Yes"])
            online_backup = st.selectbox("Cloud Storage/Online Backup", ["No", "No internet service", "Yes"])
            device_protection = st.selectbox("Hardware Device Protection Coverage", ["No", "No internet service", "Yes"])
        with col3:
            tech_support = st.selectbox("Premium 24/7 Tech Support Ticket", ["No", "No internet service", "Yes"])
            streaming_tv = st.selectbox("Entertainment - Streaming TV Bundle", ["No", "No internet service", "Yes"])
            streaming_movies = st.selectbox("Entertainment - Movies Streaming Bundle", ["No", "No internet service", "Yes"])

    with tab3:
        st.write("##")
        col1, col2 = st.columns(2)
        with col1:
            contract = st.selectbox("Contractual Agreement Terms", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Billing Interface Preference", ["No", "Yes"])
            payment_method = st.selectbox("Standard Payment Gateway System", ["Bank transfer", "Credit card", "Electronic check", "Mailed check"])
        with col2:
            tenure = st.slider("Account Lifespan / Tenure (Months)", min_value=0, max_value=72, value=24)
            monthly_charges = st.slider("Active Monthly Plan Charge Rate ($)", min_value=10.0, max_value=150.0, value=75.0, step=0.5)
            total_charges = st.number_input("Lifetime Value / Total Charges ($)", min_value=0.0, value=1500.0, step=25.0)

    # Data Processing Vector Matrix mapping
    input_data = {
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0, 'tenure': tenure, 'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges,
        'gender_Female': 1 if gender == "Female" else 0, 'gender_Male': 1 if gender == "Male" else 0,
        'Partner_No': 1 if partner == "No" else 0, 'Partner_Yes': 1 if partner == "Yes" else 0,
        'Dependents_No': 1 if dependents == "No" else 0, 'Dependents_Yes': 1 if dependents == "Yes" else 0,
        'PhoneService_No': 1 if phone_service == "No" else 0, 'PhoneService_Yes': 1 if phone_service == "Yes" else 0,
        'MultipleLines_No': 1 if multiple_lines == "No" else 0, 'MultipleLines_No phone service': 1 if multiple_lines == "No phone service" else 0, 'MultipleLines_Yes': 1 if multiple_lines == "Yes" else 0,
        'InternetService_DSL': 1 if internet_service == "DSL" else 0, 'InternetService_Fiber optic': 1 if internet_service == "Fiber optic" else 0, 'InternetService_No': 1 if internet_service == "No" else 0,
        'OnlineSecurity_No': 1 if online_security == "No" else 0, 'OnlineSecurity_No internet service': 1 if online_security == "No internet service" else 0, 'OnlineSecurity_Yes': 1 if online_security == "Yes" else 0,
        'OnlineBackup_No': 1 if online_backup == "No" else 0, 'OnlineBackup_No internet service': 1 if online_backup == "No internet service" else 0, 'OnlineBackup_Yes': 1 if online_backup == "Yes" else 0,
        'DeviceProtection_No': 1 if device_protection == "No" else 0, 'DeviceProtection_No internet service': 1 if device_protection == "No internet service" else 0, 'DeviceProtection_Yes': 1 if device_protection == "Yes" else 0,
        'TechSupport_No': 1 if tech_support == "No" else 0, 'TechSupport_No internet service': 1 if tech_support == "No internet service" else 0, 'TechSupport_Yes': 1 if tech_support == "Yes" else 0,
        'StreamingTV_No': 1 if streaming_tv == "No" else 0, 'StreamingTV_No internet service': 1 if streaming_tv == "No internet service" else 0, 'StreamingTV_Yes': 1 if streaming_tv == "Yes" else 0,
        'StreamingMovies_No': 1 if streaming_movies == "No" else 0, 'StreamingMovies_No internet service': 1 if streaming_movies == "No internet service" else 0, 'StreamingMovies_Yes': 1 if streaming_movies == "Yes" else 0,
        'Contract_Month-to-month': 1 if contract == "Month-to-month" else 0, 'Contract_One year': 1 if contract == "One year" else 0, 'Contract_Two year': 1 if contract == "Two year" else 0,
        'PaperlessBilling_No': 1 if paperless_billing == "No" else 0, 'PaperlessBilling_Yes': 1 if paperless_billing == "Yes" else 0,
        'PaymentMethod_Bank transfer': 1 if payment_method == "Bank transfer" else 0, 'PaymentMethod_Credit card': 1 if payment_method == "Credit card" else 0, 'PaymentMethod_Electronic check': 1 if payment_method == "Electronic check" else 0, 'PaymentMethod_Mailed check': 1 if payment_method == "Mailed check" else 0
    }

    features_df = pd.DataFrame([input_data])

    # Big Action Button
    st.write("##")
    if st.button("⚡ EXECUTE REAL-TIME RISK RUNTIME"):
        with st.spinner("Processing deep analysis vectors..."):
            prediction = model.predict(features_df)[0]
            probabilities = model.predict_proba(features_df)[0]
            churn_risk = probabilities[1] * 100
            
            st.write("##")
            st.markdown("### 📊 Live Analytics Matrix")
            
            # KPI Container Boxes
            c1, c2, c3 = st.columns(3)
            with c1:
                status_color = "#ef4444" if prediction == 1 else "#10b981"
                status_text = "ALERT: High Attrition Risk" if prediction == 1 else "STABLE: Healthy User"
                st.markdown(f"""
                    <div class="metric-box" style="border-top: 4px solid {status_color};">
                        <div class="metric-lbl">Diagnostic Verdict</div>
                        <div class="metric-val" style="color: {status_color}; font-size: 24px;">{status_text}</div>
                    </div>
                """, unsafe_allow_html=True)
                
            with c2:
                st.markdown(f"""
                    <div class="metric-box" style="border-top: 4px solid #3b82f6;">
                        <div class="metric-lbl">Churn Probability Matrix</div>
                        <div class="metric-val" style="color: #60a5fa;">{churn_risk:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)
                
            with c3:
                st.markdown(f"""
                    <div class="metric-box" style="border-top: 4px solid #f59e0b;">
                        <div class="metric-lbl">Account Stability Index</div>
                        <div class="metric-val" style="color: #fbbf24;">{(probabilities[0]*100):.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)

            # Interactive Progress Bar
            st.write("##")
            st.markdown("**System telemetry visualization:**")
            st.progress(float(probabilities[1]))