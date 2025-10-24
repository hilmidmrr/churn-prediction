# Churn Prediction Project

Predicting customer churn using machine learning — a full end-to-end data science project.

---

## Project Overview
This project aims to **predict customer churn** (whether a customer will leave a service) using supervised machine learning models.  
It covers the **entire data science lifecycle** — from data exploration to deployment — and serves as a complete portfolio project.

---

## Dataset
We use a public telecom customer dataset (from Kaggle).  
The dataset includes customer demographics, service usage patterns, and contract details.

**Target Variable:** `Churn`  
**Features:** gender, tenure, internetService, monthlyCharges, contractType, etc.

---

## Machine Learning Pipeline
1. **Exploratory Data Analysis (EDA)**  
   - Checked missing values and outliers  
   - Analyzed churn rates by gender, contract type, and tenure  
   - Visualized distributions using Seaborn & Matplotlib  

2. **Feature Engineering**  
   - One-hot encoding for categorical features  
   - Scaled numeric columns using StandardScaler  

3. **Model Training**  
   - Tried multiple models:
     - Logistic Regression  
     - Random Forest  
     - XGBoost  

4. **Model Evaluation**  
   - Metrics: Accuracy, Precision, Recall, F1-score  
   - Visuals: Confusion matrix & ROC curve  

---

## 📈 Model Performance
| Model | Accuracy | F1-score |
|--------|-----------|----------|
| Logistic Regression | 0.79 | 0.75 |
| Random Forest | 0.83 | 0.80 |
| XGBoost | 0.86 | 0.83 |

Final model chosen: **XGBoost**

---

## Technologies Used
- **Python 3.10**
- **pandas, numpy, scikit-learn, xgboost**
- **matplotlib, seaborn, shap**
- **streamlit** (for web deployment)
- **Jupyter Notebook** (for EDA and modeling)

---

## How to Run
Clone the repository:
```bash
git clone https://github.com/hilmidmrr/churn-prediction.git
cd churn-prediction
