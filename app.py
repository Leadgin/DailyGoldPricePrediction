import streamlit as st
import pandas as pd
import joblib
import os
import time

st.set_page_config(
    page_title="Gold Price Predictor",
    page_icon="📈",
    layout="centered"
)

# Model configuration with assigned performance metrics (MAE, RMSE, R²)
MODEL_OPTIONS = {
    "Linear Regression": {
        "file": "linear.pkl",
        "mae": 100.1905,
        "rmse": 179.0784,
        "r2": 0.9999
    },
    "XGBoost Regressor": {
        "file": "xgboost.pkl",
        "mae": 177.3516,
        "rmse": 376.6569,
        "r2": 0.9996
    },
        "Random Forest Regressor": {
        "file": "rf_regression_best.pkl",
        "mae": 147.1123,
        "rmse": 274.2819,
        "r2": 0.9998
    },
    "ElasticNet": {
        "file": "elastic_net.pkl",
        "mae": 118.8364,
        "rmse": 215.5514,
        "r2": 0.9999
    }
}

@st.cache_resource
def load_model(model_filename: str):
    """
    Safely loads a joblib serialized machine learning model with caching.
    """
    try:
        if not os.path.exists(model_filename):
            return None, f"File '{model_filename}' not found in the root directory."
        model = joblib.load(model_filename)
        return model, None
    except Exception as e:
        return None, f"Failed to load model '{model_filename}': {str(e)}"

st.sidebar.header("Model Configuration")
selected_model_name = st.sidebar.selectbox(
    "Choose Prediction Model:",
    options=list(MODEL_OPTIONS.keys()),
    index=0
)

selected_model_info = MODEL_OPTIONS[selected_model_name]
selected_model_file = selected_model_info["file"]
model_mae = selected_model_info["mae"]
model_rmse = selected_model_info["rmse"]
model_r2 = selected_model_info["r2"]

# Load selected model
model, load_error = load_model(selected_model_file)

if load_error:
    st.sidebar.error(f"❌ {load_error}")

# Display Model Metrics in sidebar
st.sidebar.markdown("---")
st.sidebar.metric(label="R² Score", value=f"{model_r2:.4f}")
st.sidebar.metric(label="MAE (Mean Absolute Error)", value=f"{model_mae:.4f}")
st.sidebar.metric(label="RMSE (Root Mean Sq Error)", value=f"{model_rmse:.4f}")

st.title("Gold Price Predictor")
st.write("Enter feature values below for Close Price Prediction.")

col1, col2 = st.columns(2)

with col1:
    open_val = st.number_input("Open Price", value=136143.0, step=100.0, format="%.2f")
    high_val = st.number_input("High Price", value=137037.0, step=100.0, format="%.2f")
    low_val = st.number_input("Low Price", value=135525.0, step=100.0, format="%.2f")

with col2:
    volume_val = st.number_input("Volume", value=51877.0, step=1000.0, min_value=0.0, format="%.2f")
    chg_val = st.number_input("Chg% (e.g. 0.02 = 2%)", value=0.02, step=0.005, format="%.4f")

validation_errors = []

# Price sanity checks
if open_val <= 0:
    validation_errors.append("Open Price must be greater than 0.")

if high_val <= 0:
    validation_errors.append("High Price must be greater than 0.")

if low_val <= 0:
    validation_errors.append("Low Price must be greater than 0.")

# Logical relationship checks
if high_val < low_val:
    validation_errors.append("High Price cannot be lower than Low Price.")

if low_val > open_val:
    validation_errors.append("Low Price cannot be higher than Open Price.")

if volume_val < 0:
    validation_errors.append("Volume cannot be negative.")

if validation_errors:
    for err in validation_errors:
        st.error(f"⚠️ Validation Error: {err}")

is_valid = len(validation_errors) == 0

if st.button("Predict Close Price", use_container_width=True, disabled=not is_valid):
    if model is not None:
        # Construct exact DataFrame matching feature order during training
        input_df = pd.DataFrame([{
            'Open': open_val,
            'High': high_val,
            'Low': low_val,
            'Volume': volume_val,
            'Chg%': chg_val
        }])

        try:
            # Start timer
            start_time = time.perf_counter()
            
            # Generate raw prediction
            raw_pred = model.predict(input_df)
            
            # End timer
            end_time = time.perf_counter()
            inference_time_ms = (end_time - start_time) * 1000

            # Extract scalar float value safely regardless of array dimensions (1D or 2D)
            prediction = float(raw_pred.item()) if hasattr(raw_pred, 'item') else float(raw_pred[0])

            st.markdown(
                f"""
                <div style="
                    background-color: #d4edda;
                    color: #155724;
                    padding: 16px;
                    margin: 8px 0px; 
                    border-radius: 8px;
                    border: 1px solid #c3e6cb;">
                    <span style="font-size: 16px;">Predicted Close Price using {selected_model_name}:</span><br>
                    <span style="font-size: 32px; font-weight: bold;">${prediction:,.2f}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.info(f"**Inference Time:** {inference_time_ms:.2f} ms")

        except Exception as pred_err:
            st.error(f"Error generating prediction: {pred_err}")
    else:
        st.error(f"Cannot generate prediction because model '{selected_model_file}' is not loaded properly.")