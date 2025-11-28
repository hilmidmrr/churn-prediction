# Telco Customer Churn Prediction
This project provides an end-to-end machine learning solution to predict customer churn in a telecommunications company.  
The goal is to identify customers who are likely to leave thr service in advance and help improve customer retention strategies.  

---

## Overview

This project is an end-to-end machine learning pipeline for predicting customer churn in the telecommunications industiry.  
It includes EDA, feature enginnering, baseline modeling, hyperparameter optimization and SHAP-based explainability.  
The main goal is to identify high-risk customers before they leave and help companies build retention strategies.  

---

## Project Structure
```
churn-prediction/
│
├── data/
│   ├── Telco-Customer-Churn.csv        # Raw dataset
│   └── cleaned_telco.csv               # Final processed dataset
│
├── models/
│   └── best_xgb_model.pkl              # Final saved XGBoost model
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_model_improvement.ipynb
│   └── 05_model_explanation.ipynb
│
├── requirements.txt
└── README.md
```

---

## 1.Explaratory Data Analysis (EDA)

Major findings from the dataset:  
* Churn rate is ~26%  
* **Month-to-month** contract customers churn much more frequently  
* **Low-tenure** customers have significantly hihger churn risk  
* **Fiber-optic** users show increased churn likelihood  
* Security-related services (OnlineSecurity, TechSupport) have strong correlation with churn  

EDA includes distribution plots, categorical breakdowns, correlatiions analysis and churn segment exploration.  

---

## 2.Feature Engineering

Transformation applied:  

* Converted categorical variables using **One-Hot Encoding**  
* Standardized numerical variables using **StandardScaler**  
* Handled missing values in **TotalCharges**  
* Removed non-informative columns (customerID)  
* Verified outliers using IQR method  

Final dataset exported as cleaned_telco.csv  

---

## 3.Modeling

Multiple models were trained:  

* Logistic Regression (balanced)  
* Random Forest  
* XGBoost (tuned with Optuna) -> best performer  

---

## 4.Model Performance

| Model          | Accuracy | Precision_Pos(Churn) | Recall_Pos(Churn) | F1_Pos(Churn) | ROC_AUC |
|----------------|----------|------------------------|--------------------|----------------|---------|
| **XGB (tuned)**     | 0.7466   | 0.5145                 | 0.8048             | 0.6277         | **0.8470** |
| **LR (balanced)**   | 0.7381   | 0.5043                 | 0.7834             | 0.6136         | 0.8414  |
| **RF (balanced)**   | 0.7864   | 0.6237                 | 0.4920             | 0.5501         | 0.8234  |

Interpretation:  
The project prioritizes high recall, since missing a churn-risk customer (false negative) is very costly for telecom companies.  
Lower precision is acceptable in churn prediction problems.  

---

## 5.Model Explainability (SHAP)

This project includes a complete explainability analysis using SHAP:

### Global Insight
- `tenure` (strong negative relationship — longer tenure = lower churn risk)
- `MonthlyCharges` (higher charges increase risk)
- `Contract_Month-to-month` (most predictive categorical variable)
- `OnlineSecurity_No`, `TechSupport_No` also strongly increase churn risk

### **Dependence Plots**
- **Tenure:** Risk drops sharply as tenure increases  
- **MonthlyCharges:** Higher charges → higher SHAP contribution to churn

### **Customer-Level Explanations**
- SHAP waterfall plots show **why a specific customer churned or stayed**
- Both churned & non-churned customer comparisons included

This makes the model highly interpretable for business stakeholders.  

---

## How to Run

### Clone the repository:
```
git clone https://github.com/hilmidmrr/churn-prediction.git
cd churn-prediction
```
### Install dependencies:
```
pip install -r requirements.txt
```
---

## Conclusion

This project a churn prediction pipeline with:
- Clean feature engineering  
- Strong model performance using tuned XGBoost  
- Full model explainability using SHAP  
- Production-ready model file (`best_xgb_model.pkl`)  

---
