import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from helpers_styling import inject_global_css
from helpers_data import get_data_path, load_data, apply_filters
from helpers_modeling import train_model, predict_batch, risk_level
from helpers_business import revenue_at_risk, roi_simulator
from helpers_charts import apply_layout

st.set_page_config(page_title="Business Impact", layout="wide")
inject_global_css()

st.title("Business Impact & Targeting")

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

# Donut: risk tiers
risk_counts = dff["risk"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0).reset_index()
risk_counts.columns = ["risk", "count"]
donut = px.pie(
    risk_counts,
    names="risk",
    values="count",
    hole=0.6,
    color="risk",
    color_discrete_map={"High": "#DC3545", "Medium": "#FFA500", "Low": "#28A745"},
)
st.plotly_chart(apply_layout(donut, "Risk Tier Distribution", height=520), use_container_width=True)

# Opportunity matrix (risk vs value proxy)
seg = dff.groupby(["Geography", "risk"]).agg(
    Customers=("Exited", "size"),
    ChurnRate=("Exited", "mean"),
    AvgValue=("ValueProxy", "mean"),
).reset_index()

opp = px.scatter(
    seg,
    x="ChurnRate",
    y="AvgValue",
    size="Customers",
    color="Geography",
    symbol="risk",
    hover_data=["Customers"],
)
opp.update_xaxes(tickformat=".0%")
st.plotly_chart(apply_layout(opp, "Segment Opportunity Matrix (Risk vs Value)", height=600), use_container_width=True)

# ROI waterfall
st.subheader("ROI Waterfall (campaign economics)")

col1, col2, col3 = st.columns(3)
threshold = col1.slider("Target High Risk threshold", 0.4, 0.9, 0.7, 0.05)
save_rate = col2.slider("Expected save rate (lift)", 0.0, 1.0, 0.25, 0.05)
offer_cost = col3.number_input("Offer cost per targeted customer", value=20.0, min_value=0.0)

rev_risk = revenue_at_risk(dff, threshold=threshold)
targeted = int((dff["churn_proba"] >= threshold).sum())
saved, cost, net, roi = roi_simulator(rev_risk, save_rate, offer_cost, targeted)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Revenue at Risk (proxy)", f"{rev_risk:,.0f}")
m2.metric("Expected Saved Value", f"{saved:,.0f}")
m3.metric("Campaign Cost", f"{cost:,.0f}")
m4.metric("Net Impact", f"{net:,.0f}", delta=f"ROI {roi:.2f}x")

wf = go.Figure(
    go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "relative", "total"],
        x=["Revenue at Risk", "Saved (lift)", "Campaign Cost", "Net Impact"],
        y=[rev_risk, saved, -cost, rev_risk + saved - cost],
        increasing={"marker": {"color": "#28A745"}},
        decreasing={"marker": {"color": "#DC3545"}},
        totals={"marker": {"color": "#0066CC"}},
    )
)
st.plotly_chart(apply_layout(wf, "Business Waterfall: Risk → Saved → Cost → Net", height=600), use_container_width=True)