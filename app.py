import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Page Settings
st.set_page_config(
    page_title="Spectral Classification",
    page_icon="🔭",
    layout="wide"
)

# Title
st.title("🔭 Spectral Line Classification")
st.write("This app classifies astronomical objects using spectral data.")

# Sidebar
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=["csv"]
)
st.sidebar.write("Example objects: Star, Galaxy, Quasar")

# Load Model
model = joblib.load("model.pkl")
# If file uploaded
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    col1, col2 = st.columns(2)

    # -------- DATA PREVIEW --------
    with col1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

    # -------- PREDICTION --------
    with col2:
        st.subheader("Prediction Result")

        prediction = model.predict(df)

        st.success(f"Predicted Object: {prediction[0]}")

    # -------- GRAPH --------
    st.subheader("Spectral Line Graph")

    fig, ax = plt.subplots()
    ax.plot(df.iloc[0])
    ax.set_xlabel("Wavelength Index")
    ax.set_ylabel("Intensity")
    ax.set_title("Spectral Intensity Plot")

    st.pyplot(fig)

else:
    st.info("Upload a CSV file from the sidebar to begin.")

# Footer
st.markdown("---")
