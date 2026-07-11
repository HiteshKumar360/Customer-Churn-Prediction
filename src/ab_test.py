import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

from src.predict import load_artifacts, predict_batch

model, feature_names = load_artifacts()

df = pd.read_csv('data/processed/telco_churn_features.csv')

TENURE_GROUP_MAP = {"New": 0, "Developing": 1, "Mature": 2, "Loyal": 3}
df['tenure_group'] = df['tenure_group'].map(TENURE_GROUP_MAP)


scored = predict_batch(df, model=model, feature_names=feature_names)

at_risk = scored[scored['churn_probability'] > 0.35].copy()
print(f"Total at-risk customers: {len(at_risk)}")

np.random.seed(42)
at_risk['group'] = np.random.choice(['control', 'treatment'], size=len(at_risk))
print(at_risk['group'].value_counts())

at_risk['adjusted_prob'] = at_risk['churn_probability']
at_risk.loc[at_risk['group'] == 'treatment', 'adjusted_prob'] *= 0.85

at_risk['churned'] = np.random.binomial(1, at_risk['adjusted_prob'])

control_rate = at_risk[at_risk['group'] == 'control']['churned'].mean()
treatment_rate = at_risk[at_risk['group'] == 'treatment']['churned'].mean()

print(f"\nControl churn rate:   {control_rate:.2%}")
print(f"Treatment churn rate: {treatment_rate:.2%}")

if control_rate > 0:
    lift = (control_rate - treatment_rate) / control_rate * 100
    print(f"Relative churn reduction: {lift:.1f}%")

table = pd.crosstab(at_risk['group'], at_risk['churned'])
chi2, p_value, dof, expected = chi2_contingency(table)

print(f"\np-value: {p_value:.4f}")
if p_value < 0.05:
    print("Result: Statistically significant difference in churn rate")
else:
    print("Result: No statistically significant difference detected")

plt.figure(figsize=(5, 4))
plt.bar(['Control', 'Treatment'], [control_rate, treatment_rate],
        color=['#888888', '#2ecc71'])
plt.ylabel('Churn Rate')
plt.title('A/B Test: Retention Offer Impact on Churn')
plt.tight_layout()
plt.savefig('screenshots/ab_test_results.png')
print("\nChart saved as 'screenshots/ab_test_results.png'")
plt.show()