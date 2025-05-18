import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("rf_sts_model.pkl")

st.title("ICL Size Predictor")
st.markdown("Enter patient biometric data to predict the recommended ICL size.")

# User input fields
wtw = st.number_input("WTW (mm)", 10.0, 14.0, 12.0, 0.01)
acd = st.number_input("ACD (mm)", 2.0, 5.0, 3.2, 0.01)
sts = st.number_input("STS (mm)", 10.0, 14.0, 11.5, 0.01)
lr = st.number_input("Lens Rise (microns)", 0, 1000, 500, 1)
method = st.selectbox("Method", ["Sonomed", "ArcScan"])

# Map method to encoded label (must match training)
method_mapping = {"ArcScan": 0, "Sonomed": 1}
method_encoded = method_mapping[method]

if st.button("Predict ICL Size"):
    input_data = np.array([[wtw, acd, sts, lr, method_encoded]])
    prediction = model.predict(input_data)
    available_sizes = [12.1, 12.6, 13.2, 13.7]
    predicted_size = available_sizes[prediction[0]]
    st.success(f"Recommended ICL Size: {predicted_size} mm")
