import streamlit as st
import pandas as pd
import os

# 1. Page Configuration - CHANGED TO "centered" TO FIX THE GIANT STRETCHING
st.set_page_config(
    page_title="XGBoost Industrial IoT Fault Diagnosis",
    page_icon="⚙️",
    layout="centered" 
)

# 2. Top-Level Header
st.title("⚙️ Predictive Maintenance in Industrial IoT")
st.markdown("### Explainable XGBoost & FFT Framework | MSc Dissertation (Grade: 81% Distinction)")
st.write("---")

# 3. Sidebar Configuration 
st.sidebar.header("🎯 Domain & Simulation Settings")
domain = st.sidebar.selectbox(
    "Select Industrial Domain Layer:",
    ["Chemical Domain (Dataset 1)", "Mechanical Domain (Dataset 2)"]
)

st.sidebar.divider()
st.sidebar.subheader("💰 Live ROI Cost Parameters")

if "Chemical" in domain:
    default_failures = 50
    default_repair = 15800  
    recall_rate = 89.87
    ai_overhead = 2400
else:
    default_failures = 40
    default_repair = 17000  
    recall_rate = 80.88
    ai_overhead = 4000

sim_failures = st.sidebar.slider("Simulated Annual Outages (Unmonitored)", 10, 150, default_failures)
sim_cost = st.sidebar.slider("Reactive Repair Cost per Asset (£)", 1000, 30000, default_repair)

# 4. Multi-Tab Architecture
tab1, tab2, tab3, tab4 = st.tabs([
    "💰 ROI Calculator", 
    "📊 Model Evaluation", 
    "🧬 Engineering & SHAP",
    "🔄 CRISP-DM Lifecycle"
])

# ==========================================
# TAB 1: DYNAMIC ROI CALCULATOR
# ==========================================
with tab1:
    st.header("💵 Financial Impact Matrix")
    st.markdown("This simulator calculates cost benefits based on your selected parameters, maintaining the model validation constraints from the dissertation data.")
    
    baseline_loss = sim_failures * sim_cost
    mitigated_failures = int(sim_failures * (recall_rate / 100))
    missed_failures = sim_failures - mitigated_failures
    active_ai_cost = (missed_failures * sim_cost) + ai_overhead
    net_savings_optimized = baseline_loss - active_ai_cost
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Reactive Strategy Cost", value=f"£{baseline_loss:,.0f}")
    with col2:
        st.metric(label="Optimized AI Cost", value=f"£{active_ai_cost:,.0f}", delta=f"-{recall_rate}% Outages")
    with col3:
        st.metric(label="Net Mitigated Savings", value=f"£{net_savings_optimized:,.0f}", delta="Verified", delta_color="normal")
        
    st.divider()
    
    st.subheader("📊 Operational Cost Comparison")
    viz_df = pd.DataFrame({
        "Maintenance Strategy": ["Traditional Reactive", "Optimized XGBoost Framework"],
        "Total Budget Loss (£)": [baseline_loss, active_ai_cost]
    })
    st.bar_chart(data=viz_df, x="Maintenance Strategy", y="Total Budget Loss (£)")

# ==========================================
# TAB 2: MODEL EVALUATION & CONFUSION MATRICES
# ==========================================
with tab2:
    st.header("📊 Scientific Model Verification Metrics")
    
    if "Chemical" in domain:
        st.subheader("🧪 Chemical Expert Model Performance")
        metrics_df = pd.DataFrame({
            "Evaluation Metric": ["Accuracy", "Recall (Catch Rate)", "F1-Score", "CV Stability"],
            "Baseline Architecture": ["91.53%", "91.14%", "---", "89.54%"],
            "Optimized Expert Model": ["91.01%", "89.87%", "89.31%", "90.48%"]
        })
        st.table(metrics_df)
        st.divider()
        st.subheader("🔢 Real Confusion Matrix (Chemical)")
        if os.path.exists("image/cm_chemical.png"):
            st.image("image/cm_chemical.png", caption="Real Confusion Matrix output from XGBoost Chemical Model")
            
    else:
        st.subheader("⚙️ Mechanical Expert Model Performance")
        metrics_df = pd.DataFrame({
            "Evaluation Metric": ["Accuracy", "Recall (Catch Rate)", "F1-Score", "CV Stability"],
            "Baseline Architecture": ["~98.60%", "69.12%", "---", "98.44%"],
            "Optimized Expert Model": ["~98.60%", "80.88%", "79.71%", "98.05%"]
        })
        st.table(metrics_df)
        st.divider()
        st.subheader("🔢 Real Confusion Matrix (Mechanical)")
        if os.path.exists("image/cm_mechanical.png"):
            st.image("image/cm_mechanical.png", caption="Real Confusion Matrix output from XGBoost Mechanical Model")

# ==========================================
# TAB 3: PHYSICS-INFORMED FEATURE ENGINEERING & SHAP
# ==========================================
with tab3:
    st.header("🧬 Feature Engineering & Explainable AI (XAI)")
    
    st.subheader("📐 Custom Feature Generation Formulas")
    if "Chemical" in domain:
        st.info("""
        **Dual-Domain Transformation Breakdown:**
        * **Time-Domain Engineering:** Extracted rolling statistical metrics including standard deviations (`RP_std`) and moving means to capture early trend drift.
        * **Frequency-Domain Signal Processing:** Applied **Fast Fourier Transform (FFT)** algorithms directly onto raw ambient properties (`VOC_FFT`) to reveal periodic volatility signatures invisible to standard classifiers.
        """)
        st.divider()
        st.subheader("🔍 Real SHAP Beeswarm Plot (Chemical Drivers)")
        if os.path.exists("image/shap_chemical.png"):
            st.image("image/shap_chemical.png", caption="True SHAP values showing VOC impact on model output")

    else:
        st.info("""
        **Domain Boundaries & Mathematical Constraints:**
        1. **Mechanical Power Feature Extraction:**
        $$Power = Torque(Nm) \\times \\left[Speed(RPM) \\times \\frac{2\\pi}{60}\\right]$$
        2. **Thermal Dissipation Delta Calculation:**
        $$\\Delta T = Process\\_Temp(K) - Air\\_Temp(K)$$
        3. **Rotational Waveforms:** Applied **FFT transformation sequences** (`Tool_wear_min_FFT`) to capture subtle mechanical tool imbalances.
        """)
        st.divider()
        st.subheader("🔍 Real SHAP Beeswarm Plot (Mechanical Drivers)")
        if os.path.exists("image/shap_mechanical.png"):
            st.image("image/shap_mechanical.png", caption="True SHAP values showing Tool Wear and Mechanical Power impact")

# ==========================================
# TAB 4: CRISP-DM FRAMEWORK LIFECYCLE
# ==========================================
with tab4:
    st.header("🔄 Applied Data Lifecycle Methodology")
    st.markdown("This deployment pipeline strictly adheres to the industrial **CRISP-DM** lifecycle standard to guarantee reliable, production-grade deployment updates.")
    st.divider()
    
    st.subheader("🔄 Real-World CRISP-DM Implementation Cycle")
    if os.path.exists("image/crisp_dm.png"):
        st.image("image/crisp_dm.png", caption="Engineering lifecycle diagram from initial understanding to XAI deployment")
