# Groundwater Quality Risk Predictor
#### Video Demo: <URL HERE — paste your YouTube link after recording>
#### Description:

Groundwater Quality Risk Predictor is a Python program that predicts whether
a groundwater sample is likely **Safe** or **Unsafe** for drinking, based on
six water quality parameters: pH, Total Dissolved Solids (TDS), fluoride,
nitrate, arsenic, and iron. It combines a rule-based labeling system, built
on official Indian drinking water standards, with a Random Forest machine
learning classifier, and is wrapped in an optional Streamlit web interface
so the tool can be used interactively rather than only from the command line.

## The problem this project addresses

Millions of people in India rely on groundwater for drinking, but groundwater
contamination — excess fluoride causing fluorosis, arsenic poisoning, or high
nitrate from agricultural runoff — is a well-documented public health issue
in several states. Comprehensive well-by-well laboratory testing is
expensive and slow, so this project explores whether a simple predictive
model, trained on available water quality readings, could help prioritize
which regions most urgently need proper testing. The goal is not to replace
certified lab testing, but to act as a low-cost triage tool that turns
limited testing budgets into more targeted interventions.

## How the labeling works

There is no dataset that comes pre-labeled with a simple "safe" or "unsafe"
tag, so labels are generated programmatically. Each water sample is compared
against the permissible limits defined in the Bureau of Indian Standards
IS 10500:2012 drinking water specification (for example, pH must fall
between 6.5 and 8.5, fluoride must not exceed 1.5 mg/L, and arsenic must
not exceed 0.01 mg/L). If **any single parameter** exceeds its limit, the
sample is labeled "Unsafe" — mirroring how real drinking water safety
assessments work, where one contaminant above its limit is enough to
disqualify a water source regardless of how safe the other parameters are.

## Files in this project

- **project.py** — the core program. It contains `main()`, which orchestrates
  the full pipeline (load data, clean it, label it, train a model, and save
  the trained model to disk), along with three supporting functions:
  `clean_data()` standardizes column names, converts values to numeric types,
  and fills missing readings with the column median so that no row is
  discarded unnecessarily; `is_unsafe()` takes a dictionary of water
  parameters and returns `True` or `False` based on the BIS limits described
  above; `get_primary_concern()` takes the same kind of dictionary and
  returns a human-readable string naming which specific parameter(s) caused
  a sample to be flagged, which makes the tool's predictions explainable
  rather than an opaque yes/no answer. A fourth function, `train_model()`,
  encodes the categorical state/district columns, trains a Random Forest
  Classifier with balanced class weights (since unsafe samples are typically
  a minority in real data), and returns the trained model along with its
  test-set accuracy.
- **test_project.py** — pytest tests for `clean_data()`, `is_unsafe()`, and
  `get_primary_concern()`, checking both straightforward safe/unsafe cases
  and edge cases like multiple simultaneous violations or missing data.
- **generate_sample_data.py** — generates a realistic synthetic groundwater
  dataset so the whole pipeline can be tested immediately without waiting on
  an external data download; regional biases (e.g. elevated fluoride in
  Rajasthan, elevated arsenic in West Bengal/Bihar) are built in to loosely
  mirror real documented contamination patterns.
- **app.py** — an optional Streamlit web interface that reuses `is_unsafe()`
  and `get_primary_concern()` directly from project.py rather than
  duplicating logic. Users can either select an existing state/district to
  see its historical risk, or manually enter their own water test results
  for a custom prediction.
- **requirements.txt** — lists all pip-installable dependencies.

## Design decisions worth explaining

I chose a Random Forest Classifier over a simpler model like logistic
regression because contamination risk does not scale linearly with each
parameter — BIS limits create hard thresholds, and tree-based models
naturally capture that kind of step-function behavior. Random Forest also
provides feature importance scores, which let the project explain *why* a
region is flagged (in testing, arsenic and nitrate levels were consistently
the strongest predictors), and it requires no feature scaling despite the
very different numeric ranges across parameters (pH is roughly 0-14, TDS is
in the hundreds or thousands, and arsenic is measured in fractions of a
milligram per liter).

I deliberately separated the pure logic functions (`is_unsafe`,
`get_primary_concern`, `clean_data`) from the model training and the web
interface, so that the core BIS-comparison logic could be tested directly
and reliably with pytest, independent of whether a full model training run
succeeds. This also means the same three functions power both the
command-line pipeline in `main()` and the Streamlit app, avoiding
duplicated logic between the two interfaces.

## How to run it

```
pip install -r requirements.txt
python generate_sample_data.py
python project.py
streamlit run app.py
```
