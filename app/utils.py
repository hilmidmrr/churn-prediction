"""
Utility functions for preprocessing and model prediction
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import os
import streamlit as st

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join("models", "best_xgb_model.pkl")
RAW_DATA_PATH = os.path.join("data", "Telco-Customer-Churn.csv")
CLEANED_DATA_PATH = os.path.join("data", "cleaned_telco.csv")

def load_model():
    """Load the trained XGBoost model using session state for caching"""
    # Use session state instead of @st.cache_resource to avoid coroutine warnings
    if 'xgb_model' not in st.session_state:
        try:
            st.session_state.xgb_model = joblib.load(MODEL_PATH)
        except Exception as e:
            st.error(f"Model yüklenirken hata: {str(e)}")
            return None
    return st.session_state.xgb_model

def load_data():
    """Load the cleaned dataset using session state for caching"""
    # Use session state instead of @st.cache_data to avoid coroutine warnings
    if 'cleaned_data' not in st.session_state:
        try:
            st.session_state.cleaned_data = pd.read_csv(CLEANED_DATA_PATH)
        except Exception as e:
            st.error(f"Temizlenmiş veri yüklenirken hata: {str(e)}")
            return None
    return st.session_state.cleaned_data

def load_raw_data():
    """Load the raw dataset for visualization using session state for caching"""
    # Use session state instead of @st.cache_data to avoid coroutine warnings
    if 'raw_data' not in st.session_state:
        try:
            df = pd.read_csv(RAW_DATA_PATH)
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            median_value = df["TotalCharges"].median()
            df["TotalCharges"] = df["TotalCharges"].fillna(median_value)
            st.session_state.raw_data = df
        except Exception as e:
            st.error(f"Ham veri yüklenirken hata: {str(e)}")
            return None
    return st.session_state.raw_data

def preprocess_input(customer_data):
    """
    Preprocess customer input data to match model requirements
    
    Args:
        customer_data: dict with customer features
    
    Returns:
        numpy array ready for prediction
    """
    try:
        # Load cleaned data first to get structure
        cleaned_df = load_data()
        if cleaned_df is None:
            raise ValueError("Temizlenmiş veri yüklenemedi")
        
        feature_cols = cleaned_df.drop("Churn", axis=1).columns.tolist()
        
        # Create DataFrame from input
        df = pd.DataFrame([customer_data])
        
        # Handle TotalCharges - use median from raw data if missing
        raw_df = load_raw_data()
        if raw_df is None:
            raise ValueError("Ham veri yüklenemedi")
            
        if pd.isna(df["TotalCharges"].iloc[0]) or df["TotalCharges"].iloc[0] == 0:
            df["TotalCharges"] = raw_df["TotalCharges"].median()
        
        # Binary encoding (0/1)
        binary_mapping = {
            "gender": {"Male": 0, "Female": 1},
            "Partner": {"No": 0, "Yes": 1},
            "Dependents": {"No": 0, "Yes": 1},
            "PhoneService": {"No": 0, "Yes": 1},
            "PaperlessBilling": {"No": 0, "Yes": 1}
        }
        
        for col, mapping in binary_mapping.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
        
        # SeniorCitizen is already 0/1
        
        # One-hot encoding for categorical variables
        cat_cols = ["MultipleLines", "InternetService", "OnlineSecurity", 
                    "OnlineBackup", "DeviceProtection", "TechSupport",
                    "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"]
        
        # Create one-hot encoded columns
        df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=False, dtype=int)
        
        # Ensure all feature columns exist (set to 0 if missing)
        for col in feature_cols:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        
        # Reorder columns to match model exactly
        df_encoded = df_encoded[feature_cols]
        
        # Convert boolean columns to int (for columns like MultipleLines_No, etc.)
        bool_cols = df_encoded.select_dtypes(include=['bool']).columns
        if len(bool_cols) > 0:
            df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)
        
        # Standardize numerical columns
        num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        
        # Get raw data for scaling
        scaler = StandardScaler()
        scaler.fit(raw_df[num_cols])
        
        # Transform input numerical columns
        df_encoded[num_cols] = scaler.transform(df_encoded[num_cols])
        
        return df_encoded.values.astype("float32")
    
    except Exception as e:
        st.error(f"Veri ön işleme hatası: {str(e)}")
        raise e

def predict_churn(customer_data):
    """
    Predict churn probability for a customer
    
    Args:
        customer_data: dict with customer features
    
    Returns:
        dict with prediction and probability
    """
    try:
        model = load_model()
        if model is None:
            raise ValueError("Model yüklenemedi")
        
        processed_data = preprocess_input(customer_data)
        
        proba = model.predict_proba(processed_data)[0]
        prediction = model.predict(processed_data)[0]
        
        return {
            "churn": bool(prediction),
            "churn_probability": float(proba[1]),
            "no_churn_probability": float(proba[0])
        }
    
    except Exception as e:
        st.error(f"Tahmin hatası: {str(e)}")
        raise e

def get_feature_importance():
    """Get feature importance from the model"""
    try:
        model = load_model()
        if model is None or not hasattr(model, 'feature_importances_'):
            return None
        
        cleaned_df = load_data()
        if cleaned_df is None:
            return None
        
        feature_names = cleaned_df.drop("Churn", axis=1).columns.tolist()
        importance = model.feature_importances_
        
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False).head(15)
        
        return feature_importance_df
    
    except Exception as e:
        st.warning(f"Feature importance alınamadı: {str(e)}")
        return None