# Churn Analysis Pipeline

An end-to-end machine learning pipeline for customer churn analysis, currently featuring a robust ETL module that processes raw data efficiently in chunks and loads it into a local database for downstream modeling.

## 🚀 Quick Setup

1. **Create the virtual environment:**
```bash
python -m venv .venv
```

2. **Activate the environment:**
```bash
# On Windows:
.venv\Scripts\activate

# On Linux/MacOS:
source .venv/bin/activate
```

3. **Install Poetry and dependencies:**
```bash
pip install poetry
poetry install
```

4. **Set up Jupyter hook:**
```bash
nbstripout --install
```

5. **Running the Project:**
```bash
python -m churn_analysis
```