# 💧 Groundwater Quality Risk Predictor
#### Description:

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![Pytest](https://img.shields.io/badge/Tested%20with-pytest-0A9EDC?logo=pytest&logoColor=white)
![License](https://img.shields.io/badge/BIS-IS%2010500%3A2012-1f6feb)

> A machine learning tool that predicts whether groundwater is **Safe** or **Unsafe** to drink, using real government water-quality data and official Indian drinking water standards (BIS IS 10500:2012). Built as part of the **IBM Project-Based Experiential Learning (PBEL)** program.

---

## 📖 Overview

Groundwater Quality Risk Predictor is a Python program that predicts whether a groundwater sample is likely **Safe** or **Unsafe** for drinking, based on water quality parameters such as pH, Total Dissolved Solids (TDS), fluoride, nitrate, chloride, and sulfate. It combines a rule-based labeling system — built on official Indian drinking water standards — with a Random Forest machine learning classifier, and is wrapped in an optional Streamlit web interface so the tool can be used interactively rather than only from the command line.

The project is intentionally **dataset-flexible**: different groundwater datasets name their columns differently (`F`, `F -`, or `Fluoride` all mean fluoride) and don't always include the same parameters. Rather than hardcoding one fixed set of columns, the program recognizes common naming variants automatically and simply skips any parameter that isn't present — so a new dataset can usually be dropped in and used without editing any code.

---

## 🎯 The Problem

Millions of people in India rely on groundwater for drinking, but groundwater contamination — excess fluoride causing fluorosis, high chloride/sulfate from industrial or agricultural runoff, or elevated nitrate — is a well-documented public health issue in several states. Comprehensive well-by-well laboratory testing is expensive and slow.

This project explores whether a lightweight predictive model, trained on publicly available water quality readings, can help **prioritize which districts most urgently need proper testing** — turning limited testing budgets into targeted interventions rather than random sampling. This directly reflects the social-impact focus of the IBM PBEL program: applying technical skills to a real, community-relevant problem.

---

## 🗃️ The Dataset

This project uses real post-monsoon groundwater quality readings for **Telangana state**, collected across three years (2018, 2019, 2020), sourced from the Central Ground Water Board via a public Kaggle dataset.

| | |
|---|---|
| **Rows** | 1,106 water quality readings |
| **Coverage** | 33 districts across Telangana |
| **Years** | 2018 – 2020 (post-monsoon) |
| **Parameters used** | pH, TDS, Fluoride, Nitrate, Chloride, Sulfate |

The three yearly files had inconsistent column naming (fluoride appeared as `F` in one file and `F -` in another) — the program's alias-matching in `clean_data()` handles this automatically.

---

## ⚖️ How Risk Is Determined

Each sample is checked against limits from the **Bureau of Indian Standards IS 10500:2012** drinking water specification. If **any single parameter** exceeds its limit, the sample is labeled `Unsafe` — mirroring how real safety assessments work, where one contaminant above its limit disqualifies a source.

| Parameter | Permissible Limit |
|---|---|
| pH | 6.5 – 8.5 |
| TDS | ≤ 2000 mg/L |
| Fluoride | ≤ 1.5 mg/L |
| Nitrate | ≤ 45 mg/L |
| Chloride | ≤ 1000 mg/L |
| Sulfate | ≤ 400 mg/L |
| Arsenic *(if present)* | ≤ 0.01 mg/L |
| Iron *(if present)* | ≤ 1.0 mg/L |

> **Note on accuracy:** because the model's input features are the same parameters used to generate the labels, the trained Random Forest achieves very high accuracy — it's effectively learning the same threshold rule used for labeling, not discovering a hidden pattern. The value of the model lies in packaging that rule into a fast, queryable, explainable tool, and in its feature-importance output, which shows which contaminants drive risk most often.

---

## 📂 Files in This Project

- **`project.py`** — the core program. It contains `main()`, which orchestrates the full pipeline (load data, clean it, label it, train a model, and save the trained model to disk), along with the required supporting functions:
  - `clean_data()` — standardizes column names by matching them against known water-quality parameter aliases (so datasets with different naming conventions, like `F` vs `Fluoride`, all map to the same standard names), fills missing numeric values with the column median, and drops rows missing state or district. Any parameter not present in a given dataset is simply left out rather than causing an error.
  - `is_unsafe()` — takes a dictionary of water parameters (any subset of the known BIS parameters) and returns `True` or `False` based on whether any parameter *present* in that dictionary exceeds its BIS limit. Missing parameters are skipped, not treated as violations.
  - `get_primary_concern()` — takes the same kind of dictionary and returns a human-readable string naming which specific parameter(s) caused a sample to be flagged, making the tool's predictions explainable rather than an opaque yes/no answer.
  - `train_model()` — encodes the categorical state/district columns, dynamically selects whichever parameter columns are present as features, trains a Random Forest Classifier with balanced class weights, and returns the trained model along with its test accuracy.
- **`test_project.py`** — pytest tests for `clean_data()`, `is_unsafe()`, and `get_primary_concern()`, covering straightforward safe/unsafe cases, multiple simultaneous violations, datasets with only a partial set of parameters, and non-standard column naming.
- **`app.py`** — an optional Streamlit web interface that reuses `is_unsafe()` and `get_primary_concern()` directly from `project.py` rather than duplicating logic. It dynamically builds its input form based on whichever parameters the trained model actually used, so it adapts automatically to whatever dataset was fed into `project.py`. Users can either select an existing district to see its historical average risk, or manually enter their own water test results for a custom prediction.
- **`generate_sample_data.py`** — generates a synthetic groundwater dataset as a fallback, useful for quickly testing the pipeline's logic before a real dataset is available.
- **`requirements.txt`** — lists all pip-installable dependencies needed to run the project.
- **`data/raw/groundwater_sample.csv`** — the combined, standardized real dataset (2018–2020 post-monsoon readings, 33 Telangana districts).

```
groundwater-predictor/
├── project.py                      # main(), clean_data(), is_unsafe(),
│                                    # get_primary_concern(), train_model()
├── test_project.py                 # pytest tests for the 3 required functions
├── app.py                          # optional Streamlit web interface
├── generate_sample_data.py         # synthetic data fallback for quick testing
├── requirements.txt
├── data/
│   ├── raw/groundwater_sample.csv      # combined, standardized dataset
│   └── cleaned/                        # auto-generated after running project.py
└── model/                              # auto-generated model + encoders
```

---

## 🧠 Design Decisions

**Why Random Forest?** Contamination risk doesn't scale linearly with each parameter — BIS limits create hard thresholds, and tree-based models capture that step-function behavior naturally. Random Forest also provides feature-importance scores (explainability) and needs no feature scaling despite wildly different parameter ranges (pH ~0–14 vs. TDS in the thousands).

**Why separate pure functions from the pipeline?** `is_unsafe()`, `get_primary_concern()`, and `clean_data()` are deliberately pure and dependency-free, so they can be tested directly and reliably with pytest — independent of whether a full model-training run succeeds. The same three functions power both the CLI pipeline (`main()`) and the Streamlit app, avoiding duplicated logic.

**Why alias-based column matching?** So the exact same codebase can ingest a completely different dataset (different state, different available parameters, different column-naming conventions) without any code changes — only the raw CSV needs to change.

---

## 🚀 How to Run

```bash
pip install -r requirements.txt

# Train the model (uses data/raw/groundwater_sample.csv)
python project.py

# Run the tests
python -m pytest test_project.py -v

# Launch the interactive web app
streamlit run app.py
```

Don't have a real dataset handy? Run `python generate_sample_data.py` first to generate a synthetic one for testing the pipeline end-to-end.

---

## ✅ Tests

```bash
$ python -m pytest test_project.py -v

test_project.py::test_is_unsafe PASSED
test_project.py::test_get_primary_concern PASSED
test_project.py::test_clean_data PASSED
```

Tests cover straightforward safe/unsafe cases, multiple simultaneous violations, missing/partial parameters, and non-standard column naming.

---

## 🌍 Relevance to IBM PBEL

This project was developed as part of the **IBM Project-Based Experiential Learning (PBEL)** program, which emphasizes applying technical skills to solve real, socially relevant problems. It demonstrates:
- Use of **real government open data** (CGWB, via Kaggle) rather than synthetic/toy datasets
- A clear **social impact narrative** — helping prioritize limited water-testing resources toward high-risk communities
- End-to-end engineering: data cleaning, machine learning, automated testing, and a usable interface — not just a model in a notebook