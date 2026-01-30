import streamlit as st
import plotly.express as px

from helpers_styling import inject_global_css
from helpers_data import get_data_path, load_data, apply_filters
from helpers_modeling import train_model, predict_batch, risk_level
from helpers_kpi import kpi_card
from helpers_charts import apply_layout
from helpers_advanced_charts import sankey_customer_journey, sunburst_value_segments, pareto_churn_segments

st.set_page_config(page_title="Overview", layout="wide")
inject_global_css()

st.title("Overview (Executive)")

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

# Batch predictions (FAST)
dff["churn_proba"] = predict_batch(model, scaler, feat_cols, dff)
dff["risk"] = dff["churn_proba"].apply(risk_level)

# KPIs
total = len(dff)
churn_rate = float(dff["Exited"].mean()) if total else 0.0
active_pct = float(dff["IsActiveMember"].mean()) if total else 0.0
avg_balance = float(dff["Balance"].mean()) if total else 0.0
high_risk = int((dff["risk"] == "High").sum())

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    kpi_card("Total Customers", f"{total:,}", border_color="#0066CC")
with c2:
    kpi_card("Churn Rate", f"{churn_rate:.1%}", border_color="#DC3545")
with c3:
    kpi_card("Active %", f"{active_pct:.1%}", border_color="#28A745")
with c4:
    kpi_card("Avg Balance", f"{avg_balance:,.0f}", border_color="#17A2B8")
with c5:
    kpi_card("High Risk Count", f"{high_risk:,}", border_color="#FFA500")

st.divider()

left, right = st.columns(2)
with left:
    st.plotly_chart(sankey_customer_journey(dff), use_container_width=True)

with right:
    geo = dff.groupby("Geography")["Exited"].mean().reset_index(name="ChurnRate")
    fig = px.bar(
        geo,
        x="Geography",
        y="ChurnRate",
        text="ChurnRate",
        color="ChurnRate",
        color_continuous_scale=["#28A745", "#FFA500", "#DC3545"],
    )
    fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
    fig.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(apply_layout(fig, "Churn Rate by Geography"), use_container_width=True)

st.plotly_chart(sunburst_value_segments(dff), use_container_width=True)
st.plotly_chart(pareto_churn_segments(dff), use_container_width=True)