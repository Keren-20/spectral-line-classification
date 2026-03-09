import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

st.title("Spectral Line Classification")

# Upload dataset
uploaded_file = st.file_uploader("Upload spectral dataset CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    # Assume last column is the label
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    st.write("Feature shape:", X.shape)
    st.write("Label shape:", y.shape)

    # Plot first spectrum
    st.subheader("Example Spectrum")

    fig, ax = plt.subplots()
    ax.plot(X.iloc[0])
    ax.set_xlabel("Wavelength Index")
    ax.set_ylabel("Intensity")
    ax.set_title("First Spectrum")

    st.pyplot(fig)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(n_estimators=200)

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)

    st.subheader("Model Performance")

    report = classification_report(y_test, y_pred, output_dict=True)

    st.dataframe(pd.DataFrame(report).transpose())

    # Predict single spectrum
    st.subheader("Predict First Spectrum")

    if st.button("Predict"):

        prediction = model.predict([X.iloc[0]])

        st.success(f"Predicted Class: {prediction[0]}")
