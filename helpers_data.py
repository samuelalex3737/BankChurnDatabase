from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st


DEFAULT_CSV_NAME = "Bank Customer Churn Prediction.csv"


def get_data_path() -> Path:
    # Repo root is current working directory on Streamlit Cloud
    return Path(DEFAULT_CSV_NAME)


@st.cache_data
def load_data(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Normalize common Kaggle column variants
    rename_map = {
        "CustomerId": "CustomerID",
        "customer_id": "CustomerID",
        "num_products": "NumOfProducts",
        "has_card": "HasCrCard",
        "is_active": "IsActiveMember",
        "estimated_salary": "EstimatedSalary",
        "credit_score": "CreditScore",
        "geography": "Geography",
        "gender": "Gender",
        "age": "Age",
        "tenure": "Tenure",
        "balance": "Balance",
        "exited": "Exited",
        "surname": "Surname",
    }
    df.rename(columns={c: rename_map[c] for c in df.columns if c in rename_map}, inplace=True)

    required = [
        "CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
        "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}. Found columns: {list(df.columns)}")

    # Ensure types
    for c in ["Exited", "HasCrCard", "IsActiveMember", "NumOfProducts"]:
        df[c] = df[c].astype(int)

    # Age banding
    bins = [0, 25, 35, 45, 55, 65, 120]
    labels = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]
    df["AgeBand"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False)

    # Value proxy: Balance × (Tenure+1)
    df["ValueProxy"] = df["Balance"].clip(lower=0) * (df["Tenure"] + 1)

    return df


def apply_filters(
    df: pd.DataFrame,
    geos: list[str],
    age_range: tuple[int, int],
    products: list[int],
    active_member: str,
) -> pd.DataFrame:
    dff = df.copy()

    if geos:
        dff = dff[dff["Geography"].isin(geos)]

    dff = dff[(dff["Age"] >= age_range[0]) & (dff["Age"] <= age_range[1])]

    if products:
        dff = dff[dff["NumOfProducts"].isin(products)]

    if active_member != "All":
        target = 1 if active_member == "Active" else 0
        dff = dff[dff["IsActiveMember"] == target]

    return dff