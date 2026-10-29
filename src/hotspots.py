"""Coverage-aware hotspot detection along a linear corridor (chainage in km).

Why coverage-aware?  In this dataset different years cover different parts of the
highway (e.g. 2013-14 km 150-250, 2015-17 km 400-575, 2022-23 km 0-180). Raw
accident counts per km would therefore mostly reflect *how long a stretch was
observed*, not how dangerous it is.  We estimate, for each segment, the number of
years it was observed (exposure) and rank segments by an empirical-Bayes shrunk
accident rate x severity weight.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import gamma as gamma_dist
from scipy.stats import spearmanr

from .config import EARLY_BLOCK, NIGHT_BLOCKS

TIERS = ["Critical", "High", "Medium", "Low"]
TIER_COLORS = {"Critical": "#d62728", "High": "#ff7f0e", "Medium": "#f2c94c",
               "Low": "#7fb3d5", "No data": "#cfcfcf"}


# ----------------------------------------------------------------------------- coverage
def n_segments(df: pd.DataFrame, seg_km: float) -> int:
    return int(df["km"].max() // seg_km) + 1


def segment_index(km: pd.Series, seg_km: float) -> pd.Series:
    return (km // seg_km).astype(int)


def coverage_matrix(df: pd.DataFrame, seg_km: float, gap_km: float = 25.0,
                    min_records: int = 30) -> pd.DataFrame:
    """1 if a segment lies inside a surveyed stretch in that year, else 0.

    For each year, sorted chainages are split into contiguous blocks wherever the gap
    exceeds `gap_km`; blocks with fewer than `min_records` accidents are treated as
    isolated records (not a surveyed stretch).
    """
    years = sorted(df["year"].unique())
    nb = n_segments(df, seg_km)
    cov = pd.DataFrame(0, index=range(nb), columns=years, dtype=int)
    for y in years:
        k = np.sort(df.loc[df["year"] == y, "km"].to_numpy())
        if len(k) == 0:
            continue
        cuts = np.where(np.diff(k) > gap_km)[0]
        starts = np.r_[0, cuts + 1]
        ends = np.r_[cuts, len(k) - 1]
        for a, b in zip(starts, ends):
            if b - a + 1 >= min_records:
                cov.loc[int(k[a] // seg_km): int(k[b] // seg_km), y] = 1
    return cov


# ----------------------------------------------------------------------------- EB model
def fit_gamma_poisson(c, t):
    """Fit rate ~ Gamma(a, b) with counts c ~ Poisson(rate * t) by marginal likelihood."""
    c = np.asarray(c, float)
    t = np.asarray(t, float)
    if c.sum() == 0 or t.sum() == 0:
        return 1.0, 1.0
    m = c.sum() / t.sum()

    def nll(p):
        a, b = np.exp(np.clip(p, -12, 12))
        ll = (gammaln(c + a) - gammaln(a) - gammaln(c + 1)
              + a * np.log(b / (b + t)) + c * np.log(t / (b + t)))
        return -ll.sum()

    best = None
    for a0 in (0.3, 1.0, 3.0):
        res = minimize(nll, np.log([a0, a0 / max(m, 1e-9)]), method="Nelder-Mead")
        if best is None or res.fun < best.fun:
            best = res
    a, b = np.exp(np.clip(best.x, -12, 12))
    return float(a), float(b)


def segment_table(df_sel: pd.DataFrame, cov: pd.DataFrame, seg_km: float,
                  weights: dict, years: list) -> pd.DataFrame:
    """One row per segment with exposure, EB rate, severity-weighted loss and risk tier."""
    nb = len(cov)
    exposure = cov[years].sum(axis=1).to_numpy().astype(float)
    seg = segment_index(df_sel["km"], seg_km)
    w = df_sel["severity"].map(weights).fillna(1.0)
    top_class = max(weights, key=weights.get)

    n = seg.value_counts().reindex(range(nb), fill_value=0).to_numpy().astype(float)
    wsum = w.groupby(seg).sum().reindex(range(nb), fill_value=0).to_numpy()
    n_top = (df_sel["severity"] == top_class).groupby(seg).sum().reindex(range(nb), fill_value=0).to_numpy()

    obs = exposure > 0
    out = pd.DataFrame({"start_km": np.arange(nb) * seg_km}, index=range(nb))
    out["end_km"] = out["start_km"] + seg_km
    out["mid_km"] = out["start_km"] + seg_km / 2
    out["label"] = out.apply(lambda r: f"km {r.start_km:g}-{r.end_km:g}", axis=1)
    out["n"] = n.astype(int)
    out["years_obs"] = exposure.astype(int)
    out["observed"] = obs
    out["top_sev_n"] = n_top.astype(int)

    if obs.sum() == 0 or n[obs].sum() == 0:
        out["raw_rate"] = out["eb_rate"] = out["mean_weight"] = out["loss"] = np.nan
        out["p_high"] = np.nan
        out["tier"] = "No data"
        return out

    a, b = fit_gamma_poisson(n[obs], exposure[obs])
    eb = np.where(obs, (n + a) / (exposure + b), np.nan)
    w_global = wsum[obs].sum() / max(n[obs].sum(), 1)
    k = 5.0  # pseudo-observations pulling mean severity weight toward the corridor mean
    w_eb = (wsum + k * w_global) / (n + k)
    loss = eb * w_eb
    corridor_rate = n[obs].sum() / exposure[obs].sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        p_high = np.where(obs, gamma_dist.sf(2 * corridor_rate, a=n + a, scale=1.0 / (exposure + b)), np.nan)

    out["raw_rate"] = np.where(obs, n / np.where(exposure > 0, exposure, 1), np.nan)
    out["eb_rate"] = eb
    out["mean_weight"] = np.where(obs, w_eb, np.nan)
    out["loss"] = loss
    out["p_high"] = p_high

    pct = pd.Series(loss[obs]).rank(pct=True).to_numpy()
    tier = np.full(nb, "No data", dtype=object)
    tier[obs] = np.select([pct >= 0.95, pct >= 0.85, pct >= 0.60], ["Critical", "High", "Medium"], "Low")
    out["tier"] = tier
    return out


# ----------------------------------------------------------------------------- backtest
def backtest(df: pd.DataFrame, cov: pd.DataFrame, seg_km: float,
             top_frac: float = 0.2, min_segments: int = 8) -> pd.DataFrame:
    """For each year Y: rank segments using only earlier years, test on year Y.

    Only segments surveyed in Y *and* in at least one earlier year are compared, so the
    test is not distorted by the patchy coverage.
    """
    years = list(cov.columns)
    nb = len(cov)
    seg = segment_index(df["km"], seg_km)
    counts = {y: seg[df["year"] == y].value_counts().reindex(range(nb), fill_value=0).to_numpy()
              for y in years}
    rows = []
    for i, y in enumerate(years):
        earlier = years[:i]
        if not earlier:
            continue
        t = cov[earlier].sum(axis=1).to_numpy().astype(float)
        c = sum(counts[e] * cov[e].to_numpy() for e in earlier).astype(float)
        mask = (cov[y].to_numpy() == 1) & (t > 0)
        if mask.sum() < min_segments:
            continue
        a, b = fit_gamma_poisson(c[t > 0], t[t > 0])
        eb = (c[mask] + a) / (t[mask] + b)
        test = counts[y][mask]
        if test.sum() == 0:
            continue
        k = max(1, int(round(top_frac * mask.sum())))
        top = np.argsort(-eb)[:k]
        rho = spearmanr(eb, test)[0] if np.ptp(eb) > 0 and np.ptp(test) > 0 else np.nan
        used = [e for e in earlier if cov[e].to_numpy()[mask].any()]
        rows.append({
            "test_year": y,
            "ranked_using": ", ".join(str(e) for e in used),
            "segments_compared": int(mask.sum()),
            "rank_correlation": rho,
            f"top_{int(top_frac * 100)}%_capture": test[top].sum() / test.sum(),
            "random_baseline": k / mask.sum(),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- insights
def segment_insights(df_sel: pd.DataFrame, lo: float, hi: float, factors: list,
                     min_count: int = 8, min_lift: float = 1.3) -> pd.DataFrame:
    """Over-represented categories in a segment relative to the whole corridor (lift)."""
    sub = df_sel[(df_sel["km"] >= lo) & (df_sel["km"] < hi)]
    rows = []
    if len(sub) == 0:
        return pd.DataFrame(columns=["factor", "category", "count", "segment_share", "corridor_share", "lift"])
    for f in factors:
        seg_share = sub[f].value_counts(normalize=True)
        all_share = df_sel[f].value_counts(normalize=True)
        cnt = sub[f].value_counts()
        for cat, s in seg_share.items():
            lift = s / all_share.get(cat, np.nan)
            if cnt[cat] >= min_count and lift >= min_lift:
                rows.append({"factor": f, "category": cat, "count": int(cnt[cat]),
                             "segment_share": s, "corridor_share": all_share[cat], "lift": lift})
    res = pd.DataFrame(rows)
    return res.sort_values("lift", ascending=False) if len(res) else res


def action_hints(ins: pd.DataFrame, top_sev_share: float, corridor_top_share: float) -> list:
    """Label-agnostic engineering hints (time/season based). To be validated by an expert."""
    hints = []
    if len(ins):
        tb = ins[ins["factor"] == "time_block"]
        if tb["category"].isin(NIGHT_BLOCKS).any():
            hints.append("Night-time accidents are over-represented: check street/edge lighting, "
                         "retro-reflective markings, delineators and night-time speed enforcement.")
        if (tb["category"] == EARLY_BLOCK).any():
            hints.append("Early-morning over-representation: consider fatigue/visibility measures "
                         "(rumble strips, rest areas, fog-warning signage).")
        mo = ins[ins["factor"] == "month"]
        if mo["category"].isin([6, 7, 8, 9]).any():
            hints.append("Monsoon-season months are over-represented: inspect drainage, "
                         "skid resistance and wet-weather warning signs.")
    if corridor_top_share > 0 and top_sev_share >= 1.5 * corridor_top_share:
        hints.append("The share of the most severe class is well above the corridor average: "
                     "review speed management, median/barrier protection and access points.")
    hints.append("Run a field road-safety audit before acting; these are data-driven leads, not diagnoses.")
    return hints
