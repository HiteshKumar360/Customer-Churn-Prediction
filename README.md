# Customer Churn Intelligence Platform

An end-to-end machine learning project that predicts customer churn for a telecom
company, segments customers into actionable groups, and recommends targeted
retention offers — built on the Telco Customer Churn dataset.

**[Live Dashboard](https://customer-churn-prediction-d7ma2oof4j4u88e6dcyguc.streamlit.app)** 

---

## Overview

Telecom companies lose significant revenue to customer churn. This project builds
a complete pipeline — from raw data to a deployed, interactive dashboard — that:

- Predicts the probability that any given customer will churn
- Explains *why* using SHAP feature importance
- Segments all customers into 3 actionable personas using K-Means
- Recommends a specific retention offer per customer based on their segment

| Metric | Score |
|---|---|
| ROC-AUC | 0.821 |
| F1 Score (threshold = 0.35) | 0.611 |
| Churn Recall | 79% |
| Overall Accuracy | 73% |

---

## Dataset

[Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

7,043 customers · 21 features · binary churn label (26.6% churn rate)

Download the dataset from Kaggle (downloads as
`WA_Fn-UseC_-Telco-Customer-Churn.csv`) and save it as:

```
data/raw/telco_churn_raw.csv
```

---

## Project Structure

```
customer-churn/
├── app/
│   └── streamlit_app.py       
├── data/
│   ├── raw/                  
│   └── processed/               
├── models/
│   ├── lgbm_churn_model.pkl     
│   └── feature_names.pkl        
├── notebooks/
│   ├── 01_eda.ipynb              
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model.ipynb            
│   └── 04_segments.ipynb        
├── src/
│   ├── preprocessing.py          
│   ├── predict.py                
│   └── train_model.py            
├── requirements.txt
└── README.md
```

---

## Pipeline

```
Raw data → EDA → Feature Engineering → Model Training → Segmentation → Dashboard
```

**1. EDA** — Cleaned the raw dataset (fixed `TotalCharges` type bug, dropped 11
incomplete rows), then explored churn drivers across contract type, tenure,
internet service, and payment method.

**2. Feature Engineering** — Built 6 new features: `tenure_group`,
`num_services`, `is_new_customer`, `CLV`, `avg_service_cost`, `high_risk_flag`.
All 6 showed correlation > 0.15 with churn.

**3. Model Training** — Trained a LightGBM classifier, tuned hyperparameters with
RandomizedSearchCV, and The threshold was reduced to 0.35 to improve churn recall, 
prioritizing the identification of potential churners over minimizing false positives. 
Tracked all experiments with MLflow.

**4. Segmentation** — Used K-Means (K=3, selected via silhouette score) on
tenure, monthly charges, number of services, and predicted churn probability
to group customers into three personas:

| Segment | Customers | Churn Rate | Avg CLV | Action |
|---|---|---|---|---|
| Champions | 2,511 | 14.0% | $4,743 | Loyalty rewards |
| At Risk | 2,421 | 57.0% | $883 | Contract upgrade offer |
| Fence Sitters | 2,100 | 7.0% | $956 | Service bundle discount |

**5. Dashboard** — A 4-page Streamlit app: live customer churn predictor,
segment explorer with PCA visualization, model performance metrics (evaluated
on a held-out test set to avoid data leakage), and business insights.

---

## Key Findings

- **Contract type** is the strongest churn predictor — month-to-month
  customers churn 15× more often than two-year contract customers (43% vs 2.8%).
- **Fiber optic** customers churn most (41.9%) despite paying the highest
  charges — a pricing/value-perception issue.
- Churners leave with a **median tenure of 10 months** vs. 38 months for
  retained customers — the first year is the critical retention window.
- Retained customers have **66% higher CLV** than churned customers
  ($2,555 vs $1,531).
- Retaining just 30% of the **At Risk** segment (2,421 customers, 57% churn)
  protects an estimated **$366,000** in revenue.

---

## Tech Stack

`Python` `pandas` `scikit-learn` `LightGBM` `imbalanced-learn (SMOTE)` `SHAP`
`MLflow` `Streamlit` `Plotly` `joblib`

---

## Running Locally

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/customer-churn.git
cd customer-churn

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset from Kaggle and place at:
#    data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv

# 4. Run the notebooks in order (01 → 04) to regenerate processed data and the trained model, OR retrain via CLI:
python -m src.train_model

# 5. Launch the dashboard
streamlit run app/streamlit_app.py
```

---

## Author

Hitesh Kumar

- GitHub: [@HiteshKumar360](https://github.com/HiteshKumar360)
