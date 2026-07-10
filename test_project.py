"""
test_project.py
-----------------
Pytest tests for the three required custom functions in project.py:
is_unsafe, get_primary_concern, and clean_data.

Run: pytest test_project.py
"""

import pandas as pd
import numpy as np
from project import is_unsafe, get_primary_concern, clean_data


def test_is_unsafe():
    # A sample with all parameters within BIS safe limits
    safe_sample = {
        "ph": 7.0, "tds": 500, "fluoride": 0.5,
        "nitrate": 10, "arsenic": 0.005, "iron": 0.2
    }
    assert is_unsafe(safe_sample) == False

    # Fluoride exceeds the BIS limit (1.5 mg/L)
    high_fluoride = {
        "ph": 7.0, "tds": 500, "fluoride": 2.0,
        "nitrate": 10, "arsenic": 0.005, "iron": 0.2
    }
    assert is_unsafe(high_fluoride) == True

    # pH outside the safe range (6.5 - 8.5)
    bad_ph = {
        "ph": 9.2, "tds": 500, "fluoride": 0.5,
        "nitrate": 10, "arsenic": 0.005, "iron": 0.2
    }
    assert is_unsafe(bad_ph) == True

    # Arsenic exceeds the strict limit (0.01 mg/L)
    high_arsenic = {
        "ph": 7.0, "tds": 500, "fluoride": 0.5,
        "nitrate": 10, "arsenic": 0.02, "iron": 0.2
    }
    assert is_unsafe(high_arsenic) == True


def test_get_primary_concern():
    # All parameters safe -> should return "None"
    safe_sample = {
        "ph": 7.0, "tds": 500, "fluoride": 0.5,
        "nitrate": 10, "arsenic": 0.005, "iron": 0.2
    }
    assert get_primary_concern(safe_sample) == "None"

    # Only fluoride is high -> should mention only Fluoride
    high_fluoride = {
        "ph": 7.0, "tds": 500, "fluoride": 2.0,
        "nitrate": 10, "arsenic": 0.005, "iron": 0.2
    }
    assert get_primary_concern(high_fluoride) == "Fluoride"

    # Multiple issues -> both should appear in the result
    multiple_issues = {
        "ph": 7.0, "tds": 500, "fluoride": 2.0,
        "nitrate": 10, "arsenic": 0.02, "iron": 0.2
    }
    result = get_primary_concern(multiple_issues)
    assert "Fluoride" in result
    assert "Arsenic" in result


def test_clean_data():
    # Build a small raw DataFrame with missing values and messy column names
    raw_df = pd.DataFrame({
        "State": ["Rajasthan", "Bihar", "Rajasthan"],
        "District": ["Jaipur", "Patna", None],
        "pH": [7.0, np.nan, 6.8],
        "TDS": [500, 600, 700],
        "Fluoride": [1.2, 1.8, np.nan],
        "Nitrate": [20, 30, 25],
        "Arsenic": [0.005, 0.03, 0.004],
        "Iron": [0.2, 0.3, 0.25],
    })

    cleaned = clean_data(raw_df)

    # Column names should be standardized to lowercase
    assert "state" in cleaned.columns
    assert "ph" in cleaned.columns

    # Row with missing district should be dropped
    assert len(cleaned) == 2

    # No missing values should remain in numeric columns
    assert cleaned["ph"].isnull().sum() == 0
    assert cleaned["fluoride"].isnull().sum() == 0
