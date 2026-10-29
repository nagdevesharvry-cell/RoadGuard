"""Severity triage model (what-if after an accident is reported) + honest validation.

Chainage (location) is deliberately NOT a feature: the dataset covers different stretches
in different years, so a location feature would just memorise survey coverage. Location
risk is handled by the hotspot engine instead.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CAT = ["dow", "location_type", "side", "road_feature", "road_condition",
       "weather", "cause", "veh1", "veh2"]
NUM = ["hour", "month"]
FEATURES = CAT + NUM


def make_pipeline(n_estimators: int = 250, seed: int = 42) -> Pipeline:
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ("num", "passthrough", NUM),
    ])
    rf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=3,
                                class_weight="balanced_subsample", random_state=seed, n_jobs=-1)
    return Pipeline([("prep", prep), ("rf", rf)])


def _groups(df: pd.DataFrame) -> np.ndarray:
    if df["year"].nunique() >= 2:
        return df["year"].to_numpy()
    return np.arange(len(df)) % 5


def _severe_score(P, classes, severe_classes):
    idx = [classes.index(c) for c in severe_classes if c in classes]
    return P[:, idx].sum(axis=1) if idx else np.zeros(len(P))


def evaluate_leave_one_year_out(df: pd.DataFrame, severe_classes=(1, 2)) -> dict:
    """Train on all years but one, test on the held-out year (repeat for every year)."""
    classes = sorted(df["severity"].unique())
    groups = _groups(df)
    P = np.zeros((len(df), len(classes)))
    for g in np.unique(groups):
        te = groups == g
        if te.all():
            continue
        m = make_pipeline(150).fit(df.loc[~te, FEATURES], df.loc[~te, "severity"])
        proba = m.predict_proba(df.loc[te, FEATURES])
        for j, c in enumerate(m.classes_):
            P[te, classes.index(c)] = proba[:, j]
    y = df["severity"].to_numpy()
    pred = np.array(classes)[P.argmax(axis=1)]
    is_severe = np.isin(y, list(severe_classes)).astype(int)
    score = _severe_score(P, classes, severe_classes)
    auc = roc_auc_score(is_severe, score) if 0 < is_severe.sum() < len(is_severe) else np.nan
    return {
        "classes": classes,
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "auc_severe": auc,
        "majority_accuracy": pd.Series(y).value_counts(normalize=True).max(),
        "random_macro_f1": 1.0 / len(classes),
        "confusion": confusion_matrix(y, pred, labels=classes),
        "n": len(df),
    }


def permutation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Drop in macro-F1 when a factor is shuffled, measured on held-out years."""
    groups = _groups(df)
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0).split(df, groups=groups))
    m = make_pipeline(150).fit(df.iloc[tr][FEATURES], df.iloc[tr]["severity"])
    res = permutation_importance(m, df.iloc[te][FEATURES], df.iloc[te]["severity"],
                                 scoring="f1_macro", n_repeats=5, random_state=0, n_jobs=1)
    out = pd.DataFrame({"factor": FEATURES, "importance": res.importances_mean,
                        "std": res.importances_std})
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def fit_final(df: pd.DataFrame) -> Pipeline:
    return make_pipeline(n_estimators=400).fit(df[FEATURES], df["severity"])


def predict_proba(model: Pipeline, row: dict) -> pd.Series:
    X = pd.DataFrame([row])[FEATURES]
    return pd.Series(model.predict_proba(X)[0], index=model.classes_)


def sensitivity(model: Pipeline, row: dict, df: pd.DataFrame, severe_classes=(1, 2)) -> pd.DataFrame:
    """Local what-if explanation: P(severe) change when one factor is reset to its typical value."""
    classes = list(model.classes_)

    def p_severe(r):
        p = predict_proba(model, r)
        return float(sum(p.get(c, 0.0) for c in severe_classes if c in classes))

    base = p_severe(row)
    rows = []
    for f in FEATURES:
        typical = df[f].median() if f == "hour" else df[f].mode().iloc[0]
        r2 = dict(row)
        r2[f] = typical
        rows.append({"factor": f, "typical_value": typical, "your_value": row[f],
                     "effect_on_p_severe": base - p_severe(r2)})
    out = pd.DataFrame(rows)
    out["abs"] = out["effect_on_p_severe"].abs()
    return out.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)
