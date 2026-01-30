import streamlit as st


def kpi_card(label: str, value: str, border_color: str = "#0066CC", delta_text: str | None = None, delta_color: str = "#1A1A1A") -> None:
    st.markdown(
        f"""
        <div class="kpi-card" style="border-left-color:{border_color};">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          {f'<div class="kpi-delta" style="color:{delta_color};">{delta_text}</div>' if delta_text else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )