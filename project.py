"""
project.py
------------
CS50P Final Project: Groundwater Quality Risk Predictor

This program predicts whether groundwater is Safe or Unsafe for drinking
based on water quality parameters (pH, TDS, fluoride, nitrate, arsenic, iron),
using BIS (IS 10500:2012) Indian drinking water standards as the reference.

Run: python project.py
(Make sure data/raw/groundwater_sample.csv exists first - run
generate_sample_data.py if you don't have a real dataset yet.)
"""

import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib


# BIS (Bureau of Indian Standards, IS 10500:2012) permissible limits.
# A water sample is considered "Unsafe" if any parameter exceeds these.
BIS_LIMITS = {
    "ph_min": 6.5,
    "ph_max": 8.5,
    "tds": 2000,
    "fluoride": 1.5,
    "nitrate": 45,
    "arsenic": 0.01,
    "iron": 1.0,
}


def main():
    """
    Runs the full pipeline: load raw data, clean it, label each row
    Safe/Unsafe using BIS standards, train a Random Forest model,
    print accuracy, and save the trained model to disk.
    """
    raw_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/groundwater_sample.csv"

    if not os.path.exists(raw_path):
        print(f"Error: '{raw_path}' not found.")
        print("Run 'python generate_sample_data.py' first, or pass a real dataset path.")
        sys.exit(1)

    print(f"Loading data from {raw_path} ...")
    df = pd.read_csv(raw_path)
    print(f"Loaded {len(df)} rows.")

    df = clean_data(df)
    print(f"After cleaning: {len(df)} rows remain.")

    df["risk_label"] = df.apply(
        lambda row: "Unsafe" if is_unsafe(row.to_dict()) else "Safe", axis=1
    )
    df["primary_concern"] = df.apply(
        lambda row: get_primary_concern(row.to_dict()), axis=1
    )
    print(df["risk_label"].value_counts())

    os.makedirs("data/cleaned", exist_ok=True)
    df.to_csv("data/cleaned/groundwater_cleaned.csv", index=False)
    print("Cleaned data saved to data/cleaned/groundwater_cleaned.csv")

    model, encoders, accuracy = train_model(df)
    print(f"\nModel trained. Test accuracy: {accuracy:.2%}")

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/model.pkl")
    joblib.dump(encoders, "model/encoders.pkl")
    print("Model saved to model/model.pkl")


def clean_data(df):
    """
    Cleans a raw groundwater DataFrame:
    - Standardizes column names (lowercase, no spaces)
    - Converts numeric columns to numbers, filling missing values with the median
    - Drops rows missing state or district (location is required)

    Returns the cleaned DataFrame.
    """
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    required_cols = ["state", "district", "ph", "tds", "fluoride", "nitrate", "arsenic", "iron"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in dataset. Available columns: {list(df.columns)}"
            )

    numeric_cols = ["ph", "tds", "fluoride", "nitrate", "arsenic", "iron"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    df = df.dropna(subset=["state", "district"])
    return df


def is_unsafe(sample):
    """
    Takes a dict of water parameters (keys: ph, tds, fluoride, nitrate,
    arsenic, iron) and returns True if any parameter exceeds BIS
    permissible limits, otherwise False.
    """
    if sample["ph"] < BIS_LIMITS["ph_min"] or sample["ph"] > BIS_LIMITS["ph_max"]:
        return True
    if sample["tds"] > BIS_LIMITS["tds"]:
        return True
    if sample["fluoride"] > BIS_LIMITS["fluoride"]:
        return True
    if sample["nitrate"] > BIS_LIMITS["nitrate"]:
        return True
    if sample["arsenic"] > BIS_LIMITS["arsenic"]:
        return True
    if sample["iron"] > BIS_LIMITS["iron"]:
        return True
    return False


def get_primary_concern(sample):
    """
    Takes a dict of water parameters and returns a comma-separated
    string naming which parameter(s) exceed BIS limits (e.g. "Fluoride,
    Arsenic"), or "None" if all parameters are within safe limits.
    """
    issues = []
    if sample["ph"] < BIS_LIMITS["ph_min"] or sample["ph"] > BIS_LIMITS["ph_max"]:
        issues.append("pH")
    if sample["tds"] > BIS_LIMITS["tds"]:
        issues.append("TDS")
    if sample["fluoride"] > BIS_LIMITS["fluoride"]:
        issues.append("Fluoride")
    if sample["nitrate"] > BIS_LIMITS["nitrate"]:
        issues.append("Nitrate")
    if sample["arsenic"] > BIS_LIMITS["arsenic"]:
        issues.append("Arsenic")
    if sample["iron"] > BIS_LIMITS["iron"]:
        issues.append("Iron")
    return ", ".join(issues) if issues else "None"


def train_model(df):
    """
    Trains a Random Forest Classifier on the given DataFrame (which must
    already have a 'risk_label' column from is_unsafe labeling).

    Returns a tuple of (trained_model, encoders_dict, test_accuracy).
    """
    df = df.copy()
    state_encoder = LabelEncoder()
    district_encoder = LabelEncoder()

    df["state_encoded"] = state_encoder.fit_transform(df["state"])
    df["district_encoded"] = district_encoder.fit_transform(df["district"])

    feature_cols = ["state_encoded", "district_encoded", "ph", "tds",
                     "fluoride", "nitrate", "arsenic", "iron"]
    X = df[feature_cols]
    y = df["risk_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    encoders = {
        "state_encoder": state_encoder,
        "district_encoder": district_encoder,
        "feature_cols": feature_cols,
    }
    return model, encoders, accuracy


if __name__ == "__main__":
    main()
