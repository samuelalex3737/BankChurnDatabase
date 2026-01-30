import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from helpers_styling import inject_global_css
from helpers_data import get_data_path, load_data, apply_filters
from helpers_charts import apply_layout

st.set_page_config(page_title="Customer Analysis", layout="wide")
inject_global_css()

st.title("Customer Drivers & Risk Patterns")

df = load_data(get_data_path())

st.sidebar.header("Filters")
geos = st.sidebar.multiselect("Geography", sorted(df["Geography"].unique().tolist()))
age_range = st.sidebar.slider("Age Range", int(df["Age"].min()), int(df["Age"].max()), (25, 60))
products = st.sidebar.multiselect("Num of Products", sorted(df["NumOfProducts"].unique().tolist()))
active_member = st.sidebar.radio("Active Member", ["All", "Active", "Not Active"], index=0)
dff = apply_filters(df, geos, age_range, products, active_member)

# Violin: Balance by churn
v = px.violin(
    dff,
    x="Exited",
    y="Balance",
    color="Exited",
    box=True,
    points="outliers",
    color_discrete_map={0: "#28A745", 1: "#DC3545"},
)
v.update_xaxes(tickmode="array", tickvals=[0, 1], ticktext=["Retained", "Churned"])
st.plotly_chart(apply_layout(v, "Balance Distribution (Violin) by Churn"), use_container_width=True)

# Box: Age by churn & products
b = px.box(
    dff,
    x="Exited",
    y="Age",
    color="NumOfProducts",
    color_discrete_sequence=["#0066CC", "#9C27B0", "#FFA500", "#DC3545"],
)
b.update_xaxes(tickmode="array", tickvals=[0, 1], ticktext=["Retained", "Churned"])
st.plotly_chart(apply_layout(b, "Age (Box Plot) by Churn, colored by Product Count"), use_container_width=True)

# Double-axis: Age band churn + avg balance
agg = dff.groupby("AgeBand").agg(ChurnRate=("Exited", "mean"), AvgBalance=("Balance", "mean")).reset_index().dropna()
fig = go.Figure()
fig.add_bar(x=agg["AgeBand"].astype(str), y=agg["ChurnRate"], name="Churn rate", marker_color="#DC3545")
fig.add_scatter(
    x=agg["AgeBand"].astype(str),
    y=agg["AvgBalance"],
    name="Avg balance",
    yaxis="y2",
    mode="lines+markers",
    line=dict(color="#0066CC", width=4),
    marker=dict(size=10),
)
fig.update_layout(
    yaxis=dict(title="Churn rate", tickformat=".0%"),
    yaxis2=dict(title="Avg balance", overlaying="y", side="right"),
    xaxis=dict(title="Age band"),
)
st.plotly_chart(apply_layout(fig, "Age Band: Churn Rate (bars) vs Avg Balance (line)", height=560), use_container_width=True)

# Quadrant scatter
mx, my = dff["EstimatedSalary"].median(), dff["Balance"].median()
q = px.scatter(
    dff,
    x="EstimatedSalary",
    y="Balance",
    color="Exited",
    color_discrete_map={0: "#28A745", 1: "#DC3545"},
    opacity=0.75,
    hover_data=["Geography", "Age", "NumOfProducts", "IsActiveMember"],
)
q.add_vline(x=mx, line_width=3, line_dash="dash", line_color="#4A4A4A")
q.add_hline(y=my, line_width=3, line_dash="dash", line_color="#4A4A4A")
st.plotly_chart(apply_layout(q, "Quadrant: Salary vs Balance (median split)"), use_container_width=True)

# Heatmap: HasCrCard x IsActiveMember -> churn rate
hm = dff.groupby(["HasCrCard", "IsActiveMember"])["Exited"].mean().reset_index()
pivot = hm.pivot(index="HasCrCard", columns="IsActiveMember", values="Exited").fillna(0)
hfig = px.imshow(pivot, text_auto=".1%", aspect="auto", color_continuous_scale=["#28A745", "#FFA500", "#DC3545"])
hfig.update_xaxes(ticktext=["Not Active", "Active"], tickvals=[0, 1], title="Is Active Member")
hfig.update_yaxes(ticktext=["No Card", "Has Card"], tickvals=[0, 1], title="Has Credit Card")
st.plotly_chart(apply_layout(hfig, "Churn Rate Heatmap: Card Ownership × Activity", height=520), use_container_width=True)

# Correlation heatmap (numeric)
num_cols = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited",
]
corr = dff[num_cols].corr()
cfig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1)
st.plotly_chart(apply_layout(cfig, "Correlation Heatmap (numeric features)", height=650), use_container_width=True)