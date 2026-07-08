import streamlit as st
import pandas as pd

st.set_page_config(page_title="Industrial IoT Predictive Maintenance", page_icon="⚙️", layout="wide")

st.title("⚙️ Industrial IoT Predictive Maintenance Framework")
st.subheader("MSc Dissertation Validation Dashboard | Grade: 81% (Distinction)")

st.markdown("""
This interactive system demonstrates the business case for the **Explainable XGBoost + FFT** framework developed in my research. 
Adjust the variables below to see how removing the interpretability gap translates directly into operational cost reductions.
""")

st.sidebar.header("🛠️ Simulation Variables")
baseline_failures = st.sidebar.slider("Annual Baseline Failures (Unmonitored Assets)", 10, 100, 50, 5)
avg_failure_cost = st.sidebar.slider("Average Reactive Repair Cost per Failure (£)", 5000, 30000, 16000, 1000)
reduction_rate = st.sidebar.slider("Model Failure Catch Rate (Recall %)", 50, 100, 88, 1)

st.divider()

# Math Engine
total_unmanaged_cost = baseline_failures * avg_failure_cost
failures_prevented = int(baseline_failures * (reduction_rate / 100))
failures_remaining = baseline_failures - failures_prevented
total_managed_cost = (failures_remaining * avg_failure_cost) + 2400  # Including sensor/cloud maintenance overhead
net_savings = total_unmanaged_cost - total_managed_cost

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Traditional Reactive Strategy Cost", value=f"£{total_unmanaged_cost:,.0f}")
with col2:
    st.metric(label="AI-Driven Strategy Cost", value=f"£{total_managed_cost:,.0f}", delta=f"-{reduction_rate}% Outages")
with col3:
    st.metric(label="Net Saved Budget", value=f"£{net_savings:,.0f}", delta="88% Downtime Mitigated", delta_color="normal")

st.divider()

# Simple Visual Bar
chart_data = pd.DataFrame({
    "Strategy": ["Traditional Reactive", "XGBoost Framework"],
    "Total Operating Loss (£)": [total_unmanaged_cost, total_managed_cost]
})
st.bar_chart(data=chart_data, x="Strategy", y="Total Operating Loss (£)", use_container_width=True)

st.info(f"💡 **Model Validation Note:** On the chemical dataset, this XGBoost system hit an **89.87% Recall** (with `VOC` and `VOC_FFT` as critical predictors). On the mechanical dataset, it achieved an **80.88% Recall** using custom features like `Mechanical_Power` and `Tool_wear_min_FFT`. This dashboard accurately models those production savings metrics.")
