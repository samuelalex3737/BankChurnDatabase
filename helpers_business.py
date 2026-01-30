from __future__ import annotations
import pandas as pd


def revenue_at_risk(
    df: pd.DataFrame,
    p_col: str = "churn_proba",
    value_col: str = "ValueProxy",
    threshold: float = 0.70,
) -> float:
    """
    Proxy for revenue/value at risk.
    Sums expected loss for targeted customers: ValueProxy * churn_probability.
    """
    if df.empty or p_col not in df.columns or value_col not in df.columns:
        return 0.0

    high = df[df[p_col] >= threshold]
    if high.empty:
        return 0.0

    return float((high[value_col] * high[p_col]).sum())


def roi_simulator(
    revenue_risk: float,
    save_rate: float,
    offer_cost_per_cust: float,
    targeted_customers: int,
):
    """
    Simple ROI math:
    expected_saved = revenue_risk * save_rate
    campaign_cost = offer_cost_per_cust * targeted_customers
    net = expected_saved - campaign_cost
    roi = net / campaign_cost
    """
    expected_saved = float(revenue_risk) * float(save_rate)
    cost = float(offer_cost_per_cust) * int(targeted_customers)
    net = expected_saved - cost
    roi = (net / cost) if cost > 0 else 0.0
    return expected_saved, cost, net, roi