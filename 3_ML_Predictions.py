import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from helpers_styling import inject_global_css
from helpers_data import get_data_path, load_data, apply_filters
from helpers_modeling import train_model, predict_batch, predict_proba, risk_level, one_hot
from helpers_charts import apply_layout

st.set_page_config(page_title="ML Predictions", layout="wide")
inject_global_css()

st.title("ML Predictions & Explainability (SHAP)")

df = load_data(get_data_path())

# Filters
st.sidebar.header("Filters")
geos = st.sidebar.multiselect("Geography", sorted(df["Geography"].unique().tolist()))
age_range = st.sidebar.slider("Age Range", int(df["Age"].min()), int(df["Age"].max()), (25, 60))
products = st.sidebar.multiselect("Num of Products", sorted(df["NumOfProducts"].unique().tolist()))
active_member = st.sidebar.radio("Active Member", ["All", "Active", "Not Active"], index=0)
dff = apply_filters(df, geos, age_range, products, active_member).copy()

# Train model once per session
if "model_bundle" not in st.session_state:
    with st.spinner("Training model (first run only)..."):
        st.session_state["model_bundle"] = train_model(df)

model, scaler, feat_cols, auc, test_bundle, explainer = st.session_state["model_bundle"]

# Predict
dff["churn_proba"] = predict_batch(model, scaler, feat_cols, dff)
dff["risk"] = dff["churn_proba"].apply(risk_level)

# Probability distribution (violin)
v = px.violin(
    dff,
    x="Geography",
    y="churn_proba",
    color="IsActiveMember",
    box=True,
    points="outliers",
    color_discrete_map={0: "#DC3545", 1: "#28A745"},
)
st.plotly_chart(apply_layout(v, "Churn Probability Distribution by Geography (Active vs Not)"), use_container_width=True)

# 3D scatter (sample to keep it fast)
sample = dff.sample(min(1500, len(dff)), random_state=1) if len(dff) > 0 else dff
s3 = px.scatter_3d(
    sample,
    x="Age",
    y="Balance",
    z="CreditScore",
    color="churn_proba",
    color_continuous_scale=["#28A745", "#FFA500", "#DC3545"],
    opacity=0.75,
)
st.plotly_chart(apply_layout(s3, "3D: Age × Balance × CreditScore (color = churn probability)", height=700), use_container_width=True)

st.subheader("Explain one customer (SHAP Waterfall)")
cid_col = "CustomerID" if "CustomerID" in dff.columns else None

if len(dff) == 0:
    st.warning("No data after filters. Adjust filters to see predictions and SHAP explanations.")
    st.stop()

if cid_col:
    # Limit dropdown size for usability
    ids = dff[cid_col].astype(str).unique().tolist()
    cid = st.selectbox("Select CustomerID", ids[:2000])
    row = dff[dff[cid_col].astype(str) == str(cid)].head(1)
else:
    idx = st.number_input("Row index", min_value=0, max_value=len(dff) - 1, value=0)
    row = dff.iloc[int(idx): int(idx) + 1]

p = predict_proba(model, scaler, feat_cols, row)
st.write(f"Predicted churn probability: **{p:.1%}** (Risk: **{risk_level(p)}**)")

# SHAP waterfall (top 10)
X = one_hot(row)
for col in feat_cols:
    if col not in X.columns:
        X[col] = 0
X = X[feat_cols]
Xs = scaler.transform(X)

shap_values = explainer.shap_values(Xs)
base = float(explainer.expected_value)
sv = shap_values[0]

order = np.argsort(np.abs(sv))[::-1][:10]
vals = sv[order]
names = np.array(feat_cols)[order]

wf = go.Figure(
    go.Waterfall(
        name="SHAP",
        orientation="v",
        measure=["relative"] * len(vals) + ["total"],
        x=list(names) + ["Prediction"],
        y=list(vals) + [float(vals.sum() + base)],
        connector={"line": {"color": "#4A4A4A"}},
        increasing={"marker": {"color": "#DC3545"}},
        decreasing={"marker": {"color": "#28A745"}},
        totals={"marker": {"color": "#0066CC"}},
    )
)
st.plotly_chart(apply_layout(wf, "SHAP Waterfall (Top 10 drivers)", height=620), use_container_width=True)

st.subheader("What‑if Simulator")

c1, c2, c3 = st.columns(3)
age = c1.slider("Age", 18, 92, int(row["Age"].iloc[0]))
credit = c2.slider("CreditScore", 350, 850, int(row["CreditScore"].iloc[0]))
products_ = c3.slider("NumOfProducts", 1, 4, int(row["NumOfProducts"].iloc[0]))

c4, c5, c6 = st.columns(3)
balance = c4.number_input("Balance", min_value=0.0, value=float(row["Balance"].iloc[0]))
salary = c5.number_input("EstimatedSalary", min_value=0.0, value=float(row["EstimatedSalary"].iloc[0]))
active = c6.selectbox("IsActiveMember", [0, 1], index=int(row["IsActiveMember"].iloc[0]))

row2 = row.copy()
row2.loc[:, "Age"] = age
row2.loc[:, "CreditScore"] = credit
row2.loc[:, "NumOfProducts"] = products_
row2.loc[:, "Balance"] = balance
row2.loc[:, "EstimatedSalary"] = salary
row2.loc[:, "IsActiveMember"] = active

p2 = predict_proba(model, scaler, feat_cols, row2)
st.metric("New churn probability", f"{p2:.1%}", delta=f"{(p2 - p):+.1%}")