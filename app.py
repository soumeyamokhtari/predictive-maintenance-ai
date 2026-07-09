import streamlit as st
import pandas as pd

# 1. Page Configuration
st.set_page_config(
    page_title="XGBoost Industrial IoT Fault Diagnosis",
    page_icon="⚙️",
    layout="wide"
)

# 2. Top-Level Header
st.title("⚙️ Cross-Domain Predictive Maintenance in Industrial IoT")
st.markdown("### Explainable XGBoost & FFT Framework | MSc Dissertation (Grade: 81% Distinction)")
st.write("---")

# 3. Sidebar Configuration (Domain Selector & Real-Time Cost Knobs)
st.sidebar.header("🎯 Domain & Simulation Settings")
domain = st.sidebar.selectbox(
    "Select Industrial Domain Layer:",
    ["Chemical Domain (Dataset 1)", "Mechanical Domain (Dataset 2)"]
)

st.sidebar.divider()
st.sidebar.subheader("💰 Live ROI Cost Parameters")

# Setting defaults to match your exact dissertation output based on domain choice
if "Chemical" in domain:
    default_failures = 50
    default_repair = 15800  # Calculated to mirror ~£790,000 baseline
    recall_rate = 89.87
    ai_overhead = 2400
else:
    default_failures = 40
    default_repair = 17000  # Calculated to mirror ~£680,000 baseline
    recall_rate = 80.88
    ai_overhead = 4000

sim_failures = st.sidebar.slider("Simulated Annual Outages (Unmonitored)", 10, 150, default_failures)
sim_cost = st.sidebar.slider("Reactive Repair Cost per Asset (£)", 1000, 30000, default_repair)

# 4. Multi-Tab Architecture for Recruiter Navigation
tab1, tab2, tab3, tab4 = st.tabs([
    "💰 Dynamic ROI Calculator", 
    "📊 Model Evaluation & Confusion Matrices", 
    "🧬 Physics-Informed Feature Engineering & SHAP",
    "🔄 CRISP-DM Framework Lifecycle"
])

# ==========================================
# TAB 1: DYNAMIC ROI CALCULATOR
# ==========================================
with tab1:
    st.header("💵 Financial Impact Matrix & Cost Mitigation Engine")
    st.markdown("This simulator calculates cost benefits based on your selected parameters, maintaining the model validation constraints from the dissertation data.")
    
    # Mathematical Core Optimization
    baseline_loss = sim_failures * sim_cost
    mitigated_failures = int(sim_failures * (recall_rate / 100))
    missed_failures = sim_failures - mitigated_failures
    
    active_ai_cost = (missed_failures * sim_cost) + ai_overhead
    net_savings_optimized = baseline_loss - active_ai_cost
    
    # High-impact metric widgets
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Traditional Reactive Strategy Cost", value=f"£{baseline_loss:,.0f}")
    with col2:
        st.metric(label="Optimized AI Strategy Cost", value=f"£{active_ai_cost:,.0f}", delta=f"-{recall_rate}% Outages Caught")
    with col3:
        st.metric(label="Net Mitigated Budget Savings", value=f"£{net_savings_optimized:,.0f}", delta="Strategy Verified", delta_color="normal")
        
    st.divider()
    
    # Native comparative graph
    st.subheader("📊 Operational Cost Comparison Plan")
    viz_df = pd.DataFrame({
        "Maintenance Strategy": ["Traditional Reactive", "Optimized XGBoost Framework"],
        "Total Budget Loss (£)": [baseline_loss, active_ai_cost]
    })
    st.bar_chart(data=viz_df, x="Maintenance Strategy", y="Total Budget Loss (£)", use_container_width=True)

# ==========================================
# TAB 2: MODEL EVALUATION & CONFUSION MATRICES
# ==========================================
with tab2:
    st.header("📊 Scientific Model Verification Metrics")
    
    if "Chemical" in domain:
        st.subheader("🧪 Chemical Expert Model Performance (AI4I 2020 Dataset)")
        
        # Performance Indicators Table
        metrics_df = pd.DataFrame({
            "Evaluation Metric": ["Accuracy", "Recall (Catch Rate)", "F1-Score", "Cross-Validation Stability"],
            "Baseline Architecture": ["91.53%", "91.14%", "---", "89.54%"],
            "Optimized Expert Model": ["91.01%", "89.87%", "89.31%", "90.48%"]
        })
        st.table(metrics_df)
        
        st.divider()
        
        # Real Confusion Matrix Image Connection
        st.subheader("🔢 Real Confusion Matrix (Chemical Domain)")
        st.image("image/cm_chemical.png", caption="Real Confusion Matrix output from XGBoost Chemical Model", use_container_width=True)
            
    else:
        st.subheader("⚙️ Mechanical Expert Model Performance (Sensor Telemetry Data)")
        
        metrics_df = pd.DataFrame({
            "Evaluation Metric": ["Accuracy", "Recall (Catch Rate)", "F1-Score", "Cross-Validation Stability"],
            "Baseline Architecture": ["~98.60%", "69.12%", "---", "98.44%"],
            "Optimized Expert Model": ["~98.60%", "80.88%", "79.71%", "98.05%"]
        })
        st.table(metrics_df)
        
        st.divider()
        
        # Real Confusion Matrix Image Connection
        st.subheader("🔢 Real Confusion Matrix (Mechanical Domain)")
        st.image("image/cm_mechanical.png", caption="Real Confusion Matrix output from XGBoost Mechanical Model", use_container_width=True)

# ==========================================
# TAB 3: PHYSICS-INFORMED FEATURE ENGINEERING & SHAP
# ==========================================
with tab3:
    st.header("🧬 Feature Engineering Matrix & Explainable AI (XAI)")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("📐 Custom Feature Generation Formulas")
        if "Chemical" in domain:
            st.markdown("""
            ### Dual-Domain Transformation Breakdown
            * **Time-Domain Engineering:** Extracted rolling statistical metrics including standard deviations (`RP_std`) and moving means to capture early trend drift.
            * **Frequency-Domain Signal Processing:** Applied **Fast Fourier Transform (FFT)** algorithms directly onto raw ambient properties (`VOC_FFT`) to reveal periodic volatility signatures invisible to standard classifiers.
            """)
        else:
            st.markdown("""
            ### Domain Boundaries & Mathematical Constraints
            To guide optimization, raw streaming IoT variables were wrapped into custom physics-informed metrics:
            
            1. **Mechanical Power Feature Extraction:**
            $$Power = Torque(Nm) \\times \\left[Speed(RPM) \\times \\frac{2\\pi}{60}\\right]$$
            
            2. **Thermal Dissipation Delta Calculation:**
            $$\\Delta T = Process\\_Temp(K) - Air\\_Temp(K)$$
            
            3. **Rotational Waveforms:**
            Applied **FFT transformation sequences** (`Tool_wear_min_FFT`) to capture subtle mechanical tool imbalances.
            """)
            
    with col_right:
        if "Chemical" in domain:
            st.subheader("🔍 Real SHAP Beeswarm Plot (Chemical Domain Drivers)")
            st.image("image/shap_chemical.png", caption="True SHAP values showing VOC impact on model output", use_container_width=True)
        else:
            st.subheader("🔍 Real SHAP Beeswarm Plot (Mechanical Domain Drivers)")
            st.image("image/shap_mechanical.png", caption="True SHAP values showing Tool Wear and Mechanical Power impact", use_container_width=True)

# ==========================================
# TAB 4: CRISP-DM FRAMEWORK LIFECYCLE
# ==========================================
with tab4:
    st.header("🔄 Applied Data Lifecycle Methodology")
    st.markdown("This deployment pipeline strictly adheres to the industrial **CRISP-DM** lifecycle standard to guarantee reliable, production-grade deployment updates.")
    
    st.divider()
    
    # Real CRISP-DM Cycle Diagram Connection
    st.subheader("🔄 Real-World CRISP-DM Implementation Cycle")
    st.image("image/crisp_dm.png", caption="My engineering lifecycle diagram from initial understanding to XAI deployment", use_container_width=True)
