# Telco Customer Churn Prediction
End-to-end machine learning pipeline and Streamlit dashboard for predicting customer churn in a telecommunications company.  
The goal is to identify high-risk customers early and support retention strategies.

---

## Live Demo
- Streamlit App: (https://churn-prediction-fsndnkztpkzx5hr8bmawm7.streamlit.app)

---

## Overview
This repository contains:
- A complete ML workflow (EDA, feature engineering, modeling, optimization, explainability)
- A production-ready Streamlit dashboard with bilingual UI (TR/EN)
- A saved XGBoost model for inference

The dashboard preserves prediction state across language changes and uses clean, responsive visualizations.

---

## Project Structure
```
churn-prediction/
├── app/
│   ├── app.py                    # Streamlit dashboard
│   ├── translations.py           # UI translations (TR/EN)
│   └── utils.py                  # Preprocessing and model utilities
├── data/
│   ├── Telco-Customer-Churn.csv  # Raw dataset
│   └── cleaned_telco.csv         # Final processed dataset
├── models/
│   └── best_xgb_model.pkl        # Final saved XGBoost model
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_model_improvement.ipynb
│   └── 05_model_explanation.ipynb
├── requirements.txt
└── README.md
```

---

## Key Insights (EDA)
- Churn rate is around 26%.
- Month-to-month contracts churn more frequently.
- Low-tenure customers show higher churn risk.
- Fiber optic users churn more often.
- Security-related services correlate strongly with churn.

---

## Feature Engineering
- One-hot encoding for categorical features
- Standardization for numeric features
- Missing values handled in TotalCharges
- Removed non-informative columns (customerID)
- Outliers checked with IQR method

---

## Modeling
Models trained:
- Logistic Regression (balanced)
- Random Forest
- XGBoost (tuned with Optuna) as the best performer

### Model Performance (Test)
| Model           | Accuracy | Precision_Pos(Churn) | Recall_Pos(Churn) | F1_Pos(Churn) | ROC_AUC |
|----------------|----------|-----------------------|-------------------|---------------|---------|
| XGB (tuned)    | 0.7466   | 0.5145                | 0.8048            | 0.6277        | 0.8470  |
| LR (balanced)  | 0.7381   | 0.5043                | 0.7834            | 0.6136        | 0.8414  |
| RF (balanced)  | 0.7864   | 0.6237                | 0.4920            | 0.5501        | 0.8234  |

---

## Explainability (SHAP)
Key global drivers:
- Tenure (negative relationship: longer tenure = lower churn risk)
- MonthlyCharges (higher charges increase churn risk)
- Contract_Month-to-month (most predictive categorical variable)
- OnlineSecurity_No and TechSupport_No increase churn risk

---

## Streamlit Dashboard
Main features:
- Churn prediction with persisted inputs and results
- Data analysis with filters and optimized charts
- Model performance summary and feature importance
- Bilingual UI (TR/EN)

### Run the Dashboard
```bash
pip install -r requirements.txt
streamlit run app/app.py
```
The app opens at `http://localhost:8501`.

---

## Deploy to Streamlit Cloud
1. Push this repo to GitHub.
2. Go to https://streamlit.io/cloud and create a new app.
3. Select your repository and set the app file path to:
   ```
   app/app.py
   ```
4. Click Deploy.

Notes:
- `requirements.txt` is for the Streamlit app runtime dependencies.
- `requirements-dev.txt` is optional and contains notebook-only dependencies (e.g., SHAP / Optuna).

---

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Run the app: `streamlit run app/app.py`
3. Explore notebooks for detailed analysis

