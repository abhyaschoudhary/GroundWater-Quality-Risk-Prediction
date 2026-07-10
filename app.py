"""
app.py
-------
Streamlit web interface for the Groundwater Quality Risk Predictor.
This is an OPTIONAL bonus interface - it reuses the tested functions
from project.py (is_unsafe, get_primary_concern) rather than duplicating
logic, so the core CS50P submission (project.py + test_project.py)
remains self-contained while this provides a friendly UI on top.

Run: streamlit run app.py
(Run 'python project.py' first so model/model.pkl exists.)
"""

import streamlit as st
import pandas as pd
import joblib
import os

from project import is_unsafe, get_primary_concern

MODEL_PATH = "model/model.pkl"
ENCODER_PATH = "model/encoders.pkl"
CLEANED_DATA_PATH = "data/cleaned/groundwater_cleaned.csv"

st.set_page_config(page_title="Groundwater Risk Predictor", page_icon="💧", layout="centered")


@st.cache_resource
def load_model_and_encoders():
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODER_PATH)
    return model, encoders


@st.cache_data
def load_cleaned_data():
    return pd.read_csv(CLEANED_DATA_PATH)


def main():
    st.title("💧 Groundwater Quality Risk Predictor")
    st.markdown(
        "Predicts groundwater contamination risk based on BIS (IS 10500) "
        "drinking water standards, using a Random Forest model."
    )

    if not os.path.exists(MODEL_PATH):
        st.error(
            "Model not found. Run these first in your terminal:\n\n"
            "1. python generate_sample_data.py\n"
            "2. python project.py"
        )
        return

    model, encoders = load_model_and_encoders()
    df = load_cleaned_data()
    state_encoder = encoders["state_encoder"]
    district_encoder = encoders["district_encoder"]
    feature_cols = encoders["feature_cols"]

    tab1, tab2 = st.tabs(["📍 Predict by District", "🧪 Custom Water Test"])

    with tab1:
        st.subheader("Select a region to check historical risk")
        states = sorted(df["state"].unique())
        selected_state = st.selectbox("State", states)
        districts = sorted(df[df["state"] == selected_state]["district"].unique())
        selected_district = st.selectbox("District", districts)

        if st.button("Predict Risk", key="district_predict"):
            subset = df[(df["state"] == selected_state) & (df["district"] == selected_district)]
            avg_row = subset[["ph", "tds", "fluoride", "nitrate", "arsenic", "iron"]].mean()

            state_enc = state_encoder.transform([selected_state])[0]
            district_enc = district_encoder.transform([selected_district])[0]

            features = pd.DataFrame([{
                "state_encoded": state_enc, "district_encoded": district_enc,
                "ph": avg_row["ph"], "tds": avg_row["tds"], "fluoride": avg_row["fluoride"],
                "nitrate": avg_row["nitrate"], "arsenic": avg_row["arsenic"], "iron": avg_row["iron"],
            }])[feature_cols]

            prediction = model.predict(features)[0]
            probability = model.predict_proba(features).max()
            display_result(prediction, probability, avg_row.to_dict())

            st.markdown("**Historical readings for this district:**")
            st.dataframe(subset[["year", "ph", "tds", "fluoride", "nitrate", "arsenic", "iron"]])

    with tab2:
        st.subheader("Enter your own water test values")
        col1, col2 = st.columns(2)
        with col1:
            state_input = st.selectbox("State (for reference)", sorted(df["state"].unique()), key="cs")
            ph_input = st.number_input("pH", 0.0, 14.0, 7.0, 0.1)
            tds_input = st.number_input("TDS (mg/L)", 0.0, value=500.0, step=10.0)
            fluoride_input = st.number_input("Fluoride (mg/L)", 0.0, value=0.5, step=0.1)
        with col2:
            district_input = st.selectbox(
                "District (for reference)",
                sorted(df[df["state"] == state_input]["district"].unique()), key="cd"
            )
            nitrate_input = st.number_input("Nitrate (mg/L)", 0.0, value=20.0, step=1.0)
            arsenic_input = st.number_input("Arsenic (mg/L)", 0.0, value=0.005, step=0.001, format="%.4f")
            iron_input = st.number_input("Iron (mg/L)", 0.0, value=0.2, step=0.05)

        if st.button("Predict Risk", key="custom_predict"):
            sample = {
                "ph": ph_input, "tds": tds_input, "fluoride": fluoride_input,
                "nitrate": nitrate_input, "arsenic": arsenic_input, "iron": iron_input,
            }
            state_enc = state_encoder.transform([state_input])[0]
            district_enc = district_encoder.transform([district_input])[0]
            features = pd.DataFrame([{
                "state_encoded": state_enc, "district_encoded": district_enc, **sample
            }])[feature_cols]

            prediction = model.predict(features)[0]
            probability = model.predict_proba(features).max()
            display_result(prediction, probability, sample)


def display_result(prediction, probability, sample):
    """Displays the prediction result, reusing is_unsafe / get_primary_concern from project.py."""
    if prediction == "Unsafe":
        st.error(f"⚠️ Risk Level: **UNSAFE** (confidence: {probability:.0%})")
    else:
        st.success(f"✅ Risk Level: **SAFE** (confidence: {probability:.0%})")

    st.write(pd.Series(sample).to_frame(name="Value"))

    concern = get_primary_concern(sample)
    if concern != "None":
        st.warning(f"Primary concern(s): {concern}")


if __name__ == "__main__":
    main()
