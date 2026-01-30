import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from helpers_styling import inject_global_css
from helpers_data import get_data_path, load_data
from helpers_modeling import train_model
from helpers_charts import apply_layout

st.set_page_config(page_title="Model Performance", layout="wide")
inject_global_css()

st.title("Model Performance (Credibility)")

df = load_data(get_data_path())

if "model_bundle" not in st.session_state:
    with st.spinner("Training model (first run only)..."):
        st.session_state["model_bundle"] = train_model(df)

model, scaler, feat_cols, auc, (X_test_s, y_test), explainer = st.session_state["model_bundle"]

proba = model.predict_proba(X_test_s)[:, 1]
pred_default = (proba >= 0.5).astype(int)

# Confusion matrix
cm = confusion_matrix(y_test, pred_default)
cm_fig = px.imshow(cm, text_auto=True, aspect="auto", color_continuous_scale=["#E8F5E9", "#DC3545"])
cm_fig.update_xaxes(title="Predicted", tickvals=[0, 1], ticktext=["Retained", "Churn"])
cm_fig.update_yaxes(title="Actual", tickvals=[0, 1], ticktext=["Retained", "Churn"])
st.plotly_chart(apply_layout(cm_fig, "Confusion Matrix (threshold=0.50)", height=520), use_container_width=True)

# ROC
fpr, tpr, _ = roc_curve(y_test, proba)
roc = go.Figure()
roc.add_scatter(x=fpr, y=tpr, mode="lines", line=dict(color="#0066CC", width=5), name=f"ROC (AUC={auc:.3f})")
roc.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#4A4A4A", dash="dash"), name="Random")
roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
st.plotly_chart(apply_layout(roc, "ROC Curve", height=520), use_container_width=True)

# Precision-Recall
prec, rec, _ = precision_recall_curve(y_test, proba)
pr = go.Figure()
pr.add_scatter(x=rec, y=prec, mode="lines", line=dict(color="#DC3545", width=5), name="Precision–Recall")
pr.update_layout(xaxis_title="Recall", yaxis_title="Precision")
st.plotly_chart(apply_layout(pr, "Precision–Recall Curve", height=520), use_container_width=True)

# Threshold tuning + profit
st.subheader("Threshold tuning (including expected profit)")

col1, col2, col3 = st.columns(3)
value_per_churn = col1.number_input("Value lost if churn happens (proxy)", value=1000.0, min_value=0.0)
offer_cost = col2.number_input("Offer cost per targeted customer", value=20.0, min_value=0.0)
save_rate = col3.slider("Save rate if targeted (effectiveness)", 0.0, 1.0, 0.25)

ths = np.linspace(0.05, 0.95, 91)
prec_list, rec_list, f1_list, profit_list = [], [], [], []

y = y_test
for t in ths:
    pred = (proba >= t).astype(int)
    prec_list.append(precision_score(y, pred, zero_division=0))
    rec_list.append(recall_score(y, pred, zero_division=0))
    f1_list.append(f1_score(y, pred, zero_division=0))

    targeted = int(pred.sum())
    tp = int(((pred == 1) & (y == 1)).sum())
    saved = tp * value_per_churn * save_rate
    cost = targeted * offer_cost
    profit_list.append(saved - cost)

tf = go.Figure()
tf.add_scatter(x=ths, y=prec_list, name="Precision", line=dict(width=4, color="#0066CC"))
tf.add_scatter(x=ths, y=rec_list, name="Recall", line=dict(width=4, color="#28A745"))
tf.add_scatter(x=ths, y=f1_list, name="F1", line=dict(width=4, color="#9C27B0"))
tf.add_scatter(x=ths, y=profit_list, name="Expected Profit", yaxis="y2", line=dict(width=5, color="#DC3545"))

tf.update_layout(
    xaxis_title="Threshold",
    yaxis_title="Score",
    yaxis2=dict(title="Profit", overlaying="y", side="right"),
)
st.plotly_chart(apply_layout(tf, "Choose a threshold that maximizes profit (not just accuracy)", height=600), use_container_width=True)