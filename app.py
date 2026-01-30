import streamlit as st

from helpers_styling import inject_global_css
from helpers_data import get_data_path, load_data

st.set_page_config(page_title="Bank Customer Churn Dashboard", layout="wide")
inject_global_css()

st.title("Bank Customer Churn Dashboard")
st.caption("Projector-friendly, insight-only dashboard with ML + explainability (SHAP).")

data_path = get_data_path()
df = load_data(data_path)

st.subheader("Dataset snapshot")
st.dataframe(df.head(25), use_container_width=True)

st.info(
    "Use the pages in the left sidebar to navigate:\n"
    "- 1_Overview\n"
    "- 2_Customer_Analysis\n"
    "- 3_ML_Predictions\n"
    "- 4_Model_Performance\n"
    "- 5_Business_Impact"
)