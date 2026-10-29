"""Plotly figure builders used by the Streamlit app."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .config import TIME_BLOCKS
from .hotspots import TIER_COLORS, TIERS

LAYOUT = dict(
    margin=dict(l=15, r=15, t=50, b=15),
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#cbd5e1", family="'Plus Jakarta Sans', 'Inter', -apple-system, sans-serif"),
)


def fig_coverage(cov: pd.DataFrame, seg_km: float) -> go.Figure:
    x = (cov.index.to_numpy() * seg_km) + seg_km / 2
    fig = go.Figure(go.Heatmap(
        z=cov.T.to_numpy(), x=x, y=[str(c) for c in cov.columns],
        colorscale=[[0, "#eeeeee"], [1, "#2c7fb8"]], showscale=False,
        hovertemplate="year %{y}<br>km %{x:.0f}<extra></extra>", xgap=0, ygap=2))
    fig.update_layout(title="Which part of the highway was surveyed in which year (blue = covered)",
                      xaxis_title="Chainage (km)", yaxis_title="Year", height=330, **LAYOUT)
    return fig


def fig_corridor(seg: pd.DataFrame, seg_km: float) -> go.Figure:
    fig = go.Figure()
    obs = seg[seg["observed"]]
    for tier in TIERS:
        d = obs[obs["tier"] == tier]
        if d.empty:
            continue
        fig.add_bar(x=d["mid_km"], y=d["loss"], name=tier, marker_color=TIER_COLORS[tier],
                    width=seg_km * 0.92,
                    customdata=np.c_[d["label"], d["n"], d["years_obs"], d["eb_rate"], d["p_high"]],
                    hovertemplate=("<b>%{customdata[0]}</b><br>accidents: %{customdata[1]} in "
                                   "%{customdata[2]} observed yr<br>EB rate/yr: %{customdata[3]:.2f}"
                                   "<br>P(rate > 2x avg): %{customdata[4]:.0%}<br>"
                                   "loss score: %{y:.1f}<extra></extra>"))
    no = seg[~seg["observed"]]
    if len(no) and len(obs):
        fig.add_bar(x=no["mid_km"], y=[obs["loss"].max() * 0.04] * len(no), name="No data",
                    marker_color=TIER_COLORS["No data"], width=seg_km * 0.92,
                    hovertemplate="%{x:.0f} km: not surveyed in selected years<extra></extra>")
    fig.update_layout(title="Corridor risk profile: severity-weighted accident loss per segment per year",
                      xaxis_title="Chainage (km)", yaxis_title="Risk score (EB rate x severity weight)",
                      barmode="overlay", height=420, legend_title="Risk tier", **LAYOUT)
    return fig


def fig_time_profile(sub: pd.DataFrame, whole: pd.DataFrame) -> go.Figure:
    a = sub["time_block"].value_counts(normalize=True).reindex(TIME_BLOCKS, fill_value=0)
    b = whole["time_block"].value_counts(normalize=True).reindex(TIME_BLOCKS, fill_value=0)
    fig = go.Figure()
    fig.add_bar(x=TIME_BLOCKS, y=a.values, name="This segment", marker_color="#d62728")
    fig.add_bar(x=TIME_BLOCKS, y=b.values, name="Whole corridor", marker_color="#9ecae1")
    fig.update_layout(title="When accidents happen (share of accidents)", barmode="group", height=320,
                      yaxis_tickformat=".0%", **LAYOUT)
    return fig


def fig_counts(s: pd.Series, title: str, color: str = "#2c7fb8", horizontal: bool = False) -> go.Figure:
    s = s.astype(float)
    if horizontal:
        fig = px.bar(x=s.values, y=s.index.astype(str), orientation="h", color_discrete_sequence=[color])
        fig.update_yaxes(autorange="reversed")
    else:
        fig = px.bar(x=s.index.astype(str), y=s.values, color_discrete_sequence=[color])
    fig.update_layout(title=title, xaxis_title=None, yaxis_title=None, height=300, **LAYOUT)
    return fig


def fig_confusion(cm: np.ndarray, names: list) -> go.Figure:
    pct = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig = px.imshow(pct, x=names, y=names, text_auto=".0%", color_continuous_scale="Blues", aspect="auto")
    fig.update_layout(title="Confusion matrix (share of each true class), leave-one-year-out",
                      xaxis_title="Predicted", yaxis_title="Actual", coloraxis_showscale=False,
                      height=380, **LAYOUT)
    return fig


def fig_importance(imp: pd.DataFrame, names: dict) -> go.Figure:
    d = imp.copy()
    d["name"] = d["factor"].map(lambda f: names.get(f, f))
    fig = px.bar(d, x="importance", y="name", orientation="h", error_x="std",
                 color_discrete_sequence=["#2c7fb8"])
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title="Permutation importance (drop in macro-F1 on held-out years)",
                      xaxis_title="Drop in macro-F1 when shuffled", yaxis_title=None, height=380, **LAYOUT)
    return fig


def fig_backtest(bt: pd.DataFrame, cap_col: str) -> go.Figure:
    x = bt["test_year"].astype(str)
    fig = go.Figure()
    fig.add_bar(x=x, y=bt[cap_col], name="Share of next-year accidents inside predicted top segments",
                marker_color="#d62728")
    fig.add_bar(x=x, y=bt["random_baseline"], name="Random guessing", marker_color="#bbbbbb")
    fig.update_layout(title="Do past hotspots capture future accidents?", barmode="group",
                      yaxis_tickformat=".0%", height=340, legend=dict(orientation="h", y=-0.2), **LAYOUT)
    return fig


def fig_sensitivity(sens: pd.DataFrame, names: dict) -> go.Figure:
    d = sens.head(8).copy()
    d["name"] = d["factor"].map(lambda f: names.get(f, f))
    colors = ["#d62728" if v > 0 else "#2c7fb8" for v in d["effect_on_p_severe"]]
    fig = go.Figure(go.Bar(x=d["effect_on_p_severe"], y=d["name"], orientation="h", marker_color=colors))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title="What pushes this accident's severity up (red) or down (blue)",
                      xaxis_title="Change in P(severe) vs. a typical value of that factor",
                      xaxis_tickformat="+.0%", height=360, **LAYOUT)
    return fig
