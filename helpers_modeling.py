from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap


FEATURES = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary",
    "Geography", "Gender",
]


def one_hot(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].copy()
    X = pd.get_dummies(X, columns=["Geography", "Gender"], drop_first=True)
    return X


def train_model(df: pd.DataFrame, seed: int = 42):
    X = one_hot(df)
    y = df["Exited"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # Keep with_mean=False to avoid issues with sparse-ish matrices
    scaler = StandardScaler(with_mean=False)
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    sm = SMOTE(random_state=seed)
    X_res, y_res = sm.fit_resample(X_train_s, y_train)

    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=seed,
        eval_metric="logloss",
    )
    model.fit(X_res, y_res)

    proba = model.predict_proba(X_test_s)[:, 1]
    auc = roc_auc_score(y_test, proba)

    explainer = shap.TreeExplainer(model)

    feature_cols = list(X.columns)
    return model, scaler, feature_cols, auc, (X_test_s, y_test), explainer


def predict_proba(model, scaler, feature_columns: list[str], df_row: pd.DataFrame) -> float:
    X = one_hot(df_row)

    # align columns
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_columns]

    Xs = scaler.transform(X)
    return float(model.predict_proba(Xs)[:, 1][0])


def predict_batch(model, scaler, feature_columns: list[str], df: pd.DataFrame) -> np.ndarray:
    X = one_hot(df)

    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_columns]

    Xs = scaler.transform(X)
    return model.predict_proba(Xs)[:, 1]


def risk_level(p: float) -> str:
    if p >= 0.70:
        return "High"
    if p >= 0.40:
        return "Medium"
    return "Low"