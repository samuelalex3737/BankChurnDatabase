from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from helpers_charts import apply_layout


def sankey_customer_journey(df: pd.DataFrame):
    g = df["Geography"].astype(str)
    p = df["NumOfProducts"].astype(str).map(lambda x: f"{x} Products")
    a = df["IsActiveMember"].map({1: "Active", 0: "Not Active"})
    e = df["Exited"].map({1: "Churned", 0: "Retained"})

    labels = pd.Index(pd.concat([g, p, a, e], axis=0).unique())
    idx = {lab: i for i, lab in enumerate(labels)}

    def links(src_series, tgt_series):
        tmp = pd.DataFrame({"s": src_series, "t": tgt_series})
        agg = tmp.groupby(["s", "t"]).size().reset_index(name="v")
        return agg

    l1 = links(g, p)
    l2 = links(p, a)
    l3 = links(a, e)

    sources = [*l1["s"], *l2["s"], *l3["s"]]
    targets = [*l1["t"], *l2["t"], *l3["t"]]
    values = [*l1["v"], *l2["v"], *l3["v"]]

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(pad=18, thickness=22, label=[str(x) for x in labels], color="#0066CC"),
                link=dict(
                    source=[idx[s] for s in sources],
                    target=[idx[t] for t in targets],
                    value=values,
                    color="rgba(0,102,204,0.35)",
                ),
            )
        ]
    )
    return apply_layout(fig, "Customer Journey Flow: Geography → Products → Activity → Outcome", height=650)


def sunburst_value_segments(df: pd.DataFrame):
    fig = px.sunburst(
        df,
        path=["Geography", "AgeBand", "NumOfProducts"],
        values="ValueProxy",
        color="Exited",
        color_continuous_scale=["#28A745", "#DC3545"],
    )
    return apply_layout(fig, "Value Segments (ValueProxy = Balance × (Tenure+1))", height=650)


def pareto_churn_segments(df: pd.DataFrame):
    d = df.copy()
    d["segment"] = (
        d["Geography"].astype(str)
        + " | "
        + d["IsActiveMember"].map({1: "Active", 0: "Not Active"})
        + " | "
        + d["NumOfProducts"].astype(str)
        + "P"
    )

    churned = (
        d[d["Exited"] == 1]
        .groupby("segment")
        .size()
        .sort_values(ascending=False)
        .reset_index(name="ChurnedCount")
    )

    if churned.empty:
        # Avoid divide by zero errors if filters remove churned customers
        churned = pd.DataFrame({"segment": [], "ChurnedCount": [], "CumPct": []})
    else:
        churned["CumPct"] = churned["ChurnedCount"].cumsum() / churned["ChurnedCount"].sum()

    fig = go.Figure()
    fig.add_bar(x=churned["segment"], y=churned["ChurnedCount"], name="Churned customers", marker_color="#DC3545")
    fig.add_scatter(
        x=churned["segment"],
        y=churned["CumPct"],
        name="Cumulative %",
        yaxis="y2",
        mode="lines+markers",
        line=dict(color="#0066CC", width=4),
        marker=dict(size=10),
    )

    fig.update_layout(
        yaxis=dict(title="Churned customers"),
        yaxis2=dict(title="Cumulative % of churn", overlaying="y", side="right", tickformat=".0%"),
        xaxis=dict(title="Segment (sorted)"),
    )
    return apply_layout(fig, "Pareto: Which segments explain most churn?", height=620)