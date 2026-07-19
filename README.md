# Churn Analysis Pipeline

An end-to-end MLOps pipeline for customer churn prediction. This project automates data extraction,
Scikit-Learn/Imblearn feature engineering, Optuna hyperparameter tuning, and Mlflow model tracking.

# Getting Started

1. **Clone the repository:**
```bash
git clone [https://github.com/davi-ssantos1/churn-analysis.git](https://github.com/davi-ssantos1/churn-analysis.git)
cd churn_analysis
```

2. **Set up the environment (via poetry):**
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/MacOS: source .venv/bin/activate

pip install poetry
poetry install
```

3. **Set up Jupyter hooks:**
```bash
nbstripout --install
```

4. **Download the source data:**
Download the raw CSV from the [Kaglle Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
Extract the file and save it exactly as `Telco-Customer-Churn.csv` inside the `data/raw_data/` directory.

5. **Run the ML Pipeline:**
```bash
python -m churn_analysis
```

6. **View the Model Registry:**
Once the pipeline finishes, launch the tracking server to view the evaluation metrics and tuned models:
```bash
mlflow ui
```