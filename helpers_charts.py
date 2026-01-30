def apply_layout(fig, title: str | None = None, height: int = 520):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=16, color="#1A1A1A"),
        legend=dict(font=dict(size=16), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=30, r=30, t=60, b=40),
        height=height,
    )
    if title:
        fig.update_layout(title=dict(text=title, x=0.01, xanchor="left", font=dict(size=26)))
    fig.update_xaxes(title_font=dict(size=20), tickfont=dict(size=14))
    fig.update_yaxes(title_font=dict(size=20), tickfont=dict(size=14))
    return fig