# Bank Customer Churn Prediction Dashboard (Streamlit)

Projector-friendly, insight-focused dashboard for the Kaggle-style **Bank Customer Churn** dataset.
Includes executive KPIs, segmentation, advanced visualizations, ML predictions, SHAP explainability,
model performance evaluation, and a business ROI simulator.

---

## Live App
After deployment, paste your Streamlit Cloud link here:
- https://<your-app-name>.streamlit.app

---

## Key Features (What this dashboard answers)
### Executive / Business Insights
- **Where is churn concentrated?**
  - Churn by Geography
  - Pareto (80/20) churn concentration by actionable segments
  - Sankey “customer journey” flows showing dominant churn pathways

- **Which segments are high-risk AND high-value?**
  - Sunburst value segmentation (ValueProxy = Balance × (Tenure + 1))
  - Opportunity matrix (Risk vs Value proxy)

- **What levers reduce churn risk?**
  - Product & activity churn heatmaps
  - “What-if” simulator to test interventions

### Machine Learning + Explainability
- XGBoost churn model trained in-app (demo-friendly)
- **Churn probability predictions**
- **SHAP waterfall** explanation for a selected customer
- Risk tiers: **High / Medium / Low**

### Model Credibility
- ROC curve + AUC
- Precision–Recall curve
- Confusion matrix
- Threshold tuning including a profit-based view

### Business Impact
- **Waterfall**: Revenue-at-risk → saved value → campaign cost → net impact
- ROI simulator controls (threshold, save rate, offer cost)

---

## Dataset
This app expects the dataset file to exist in the repo root:

**`Bank Customer Churn Prediction.csv`**

Required columns (Kaggle format):
- `CustomerID` (optional but recommended)
- `Surname` (optional)
- `CreditScore`
- `Geography`
- `Gender`
- `Age`
- `Tenure`
- `Balance`
- `NumOfProducts`
- `HasCrCard`
- `IsActiveMember`
- `EstimatedSalary`
- `Exited` (target: 1 = churned, 0 = retained)

> The app attempts to auto-normalize some common column name variants.

