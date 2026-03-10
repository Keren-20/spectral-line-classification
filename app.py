import streamlit as st # pyright: ignore[reportMissingImports]
import pandas as pd # pyright: ignore[reportMissingModuleSource]
import numpy as np # pyright: ignore[reportMissingImports]
import joblib # type: ignore
import matplotlib.pyplot as plt # pyright: ignore[reportMissingModuleSource]

# Page configuration
st.set_page_config(
    page_title="Spectral Line Classifier",
    page_icon="🔬",
    layout="wide"
)

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Custom CSS Styling
st.markdown("""
    <style>
        .main {
            background-color: #f4f6f9;
        }
        h1 {
            color: #1f4e79;
        }
        .stButton>button {
            background-color: #1f77b4;
            color: white;
            font-size: 16px;
            border-radius: 8px;
            padding: 8px 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Title Section
st.title("🔬 Spectral Line Classification System")
st.markdown("### Machine Learning Based Atomic Spectral Analysis")

st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Application Settings")
show_plot = st.sidebar.checkbox("Show Spectral Plot", value=True)
show_data = st.sidebar.checkbox("Show Uploaded Data", value=True)

uploaded_file = st.file_uploader("📂 Upload Spectral CSV File", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    if show_data:
        st.subheader("📊 Uploaded Data Preview")
        st.dataframe(data.head())

    # Separate features
    if "label" in data.columns:
        X = data.drop("label", axis=1)
    else:
        X = data

    # Visualization Section
    if show_plot:
        st.subheader("📈 Spectral Curve (First Sample)")
        fig, ax = plt.subplots()
        ax.plot(X.iloc[0])
        ax.set_xlabel("Wavelength Index")
        ax.set_ylabel("Intensity")
        ax.set_title("Spectral Signature")
        st.pyplot(fig)

    # Prediction Section
    st.subheader("🤖 Model Prediction")

    X_scaled = scaler.transform(X)

     # Predictions
    predictions = model.predict(X_scaled)

     # Prediction probabilities (confidence)
    probabilities = model.predict_proba(X_scaled)
    confidence = (probabilities.max(axis=1) * 100).round(2)

     # Add results to dataset
    data["Predicted_Label"] = predictions
    data["Confidence (%)"] = confidence

    st.success("✅ Classification Completed Successfully!")

    st.dataframe(data.head())

    # Download Button
    csv = data.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download Predictions",
        csv,
        "spectral_predictions.csv",
        "text/csv"
    )

else:
    st.info("Please upload a CSV file to begin classification.")

st.markdown("---")
st.markdown("© 2026 Spectral Line Classification Project | B.Sc Physics")




