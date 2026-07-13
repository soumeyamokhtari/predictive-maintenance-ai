# Cross-Domain Predictive Maintenance in Industrial IoT: Comparing Time- and Frequency-Domain Features Using Explainable XGBoost

## 🏆 MSc Dissertation Project — Grade: 81% (Distinction)

### 🔗 Live Interactive Dashboard
[👉 Click here to launch the Live Dashboard](https://predictive-maintenance-ai-c9hqs5elbyo72t7drefoqz.streamlit.app/)

---

## 📈 Executive Project Summary
This research bridges the interpretability gap in industrial fault diagnosis by developing an end-to-end Machine Learning framework following the **CRISP-DM** methodology. By integrating advanced signal processing (**Fast Fourier Transform**) with physics-informed feature engineering, the framework extracts high-fidelity insights from raw multi-domain sensor telemetry across both chemical and mechanical industrial settings.

### 📊 Model Performance & Data Validation
Using **XGBoost** models optimised via **GridSearchCV** and balanced via **SMOTE**, the system achieved high-accuracy predictive thresholds:
* **Chemical Domain Expert Model:** **89.87% Recall** | **89.31% F1-Score** (Validated on AI4I 2020 Predictive Maintenance Dataset)
* **Mechanical Domain Expert Model:** **80.88% Recall** | **79.71% F1-Score** (Validated on Sensor Machine Failure Dataset)

---

## 🔍 Explainable AI (XAI) & Feature Engineering Insights
Using **SHAP (SHapley Additive exPlanations)**, the black-box nature of the XGBoost models was dismantled to expose exactly which sensor values drive machine failures:

* **Chemical System Failure Drivers:** Volatile Organic Compounds (`VOC`) proved to be the highest predictor. Engineered frequency-domain features like `VOC_FFT` and time-domain variables like `RP_std` ranked among the most critical indicators.
* **Mechanical System Failure Drivers:** Tool wear metrics (`Tool_Wear_min`) dominated failure predictions. The custom physics-informed features—`Mechanical_Power`, `Temp_Delta_std`, and `Tool_wear_min_FFT`—were successfully validated as high-ranking critical variables.

---

## 🛠️ Technical Tech Stack
* **Language:** Python 3.x
* **Machine Learning:** XGBoost, Scikit-learn, SMOTE (Imbalanced-learn)
* **Explainable AI:** SHAP
* **Signal Processing:** Fast Fourier Transform (FFT), SciPy
* **Data Engineering:** Pandas, NumPy
* **Optimisation & Visualisation:** GridSearchCV, Streamlit Cloud

---
## 💻 Core Source Code
* Standard core machine learning pipelines, evaluation metrics, and SHAP explainability visualisations can be viewed directly in the [Predictive_Maintenance.ipynb](./Predictive_Maintenance.ipynb) notebook file.
---


## 🚀 How to Run Locally
1. Clone this repository:
   ```bash
   git clone https://github.com/soumeyamokhtari/Predictive-maintenance-ai.git
