import streamlit as st

PRIMARY_BG = "#FFFFFF"
SECONDARY_BG = "#F8F9FA"
TEXT_PRIMARY = "#1A1A1A"
TEXT_SECONDARY = "#4A4A4A"

RISK_COLORS = {"High": "#DC3545", "Medium": "#FFA500", "Low": "#28A745"}

PLOTLY_COLORS = ['#0066CC', '#DC3545', '#28A745', '#FFA500', '#9C27B0', '#00BCD4', '#FF5722', '#4CAF50']


def inject_global_css() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {PRIMARY_BG};
            color: {TEXT_PRIMARY};
        }}

        html, body, [class*="css"] {{
            font-size: 18px !important;
        }}

        h1 {{
            font-size: 48px !important;
            font-weight: 800 !important;
            color: {TEXT_PRIMARY} !important;
        }}
        h2 {{
            font-size: 36px !important;
            font-weight: 800 !important;
            color: #2C3E50 !important;
            margin-top: 14px !important;
        }}
        h3 {{
            font-size: 26px !important;
            font-weight: 700 !important;
        }}

        section[data-testid="stSidebar"] {{
            background: {SECONDARY_BG};
        }}
        section[data-testid="stSidebar"] * {{
            font-size: 18px !important;
        }}

        .plot-container {{
            border: 2px solid #EAEAEA;
            border-radius: 12px;
            padding: 6px;
        }}

        .kpi-card {{
            background: {PRIMARY_BG};
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            padding: 18px 18px 14px 18px;
            min-height: 180px;
            border-left: 8px solid #0066CC;
        }}
        .kpi-label {{
            color: {TEXT_SECONDARY};
            font-size: 20px;
            font-weight: 600;
        }}
        .kpi-value {{
            color: {TEXT_PRIMARY};
            font-size: 42px;
            font-weight: 900;
            line-height: 1.1;
        }}
        .kpi-delta {{
            font-size: 18px;
            font-weight: 700;
            margin-top: 6px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )