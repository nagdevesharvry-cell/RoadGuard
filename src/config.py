"""Central configuration: paths, label dictionaries, severity weights, time blocks.

Edit `data/labels.csv` (columns: column,code,label) to give the numeric codes in the
dataset real names. Anything not listed there is shown as "Code <n>".
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DEFAULT_CSV = DATA_DIR / "ETP_4_New_Data_Accidents.csv"
LABELS_CSV = DATA_DIR / "labels.csv"

# Severity weights used for the hotspot "loss" score (editable in the sidebar).
# ASSUMPTION: 1 = most severe (fatal) ... 4 = least severe. Change if your data
# dictionary says otherwise (and update labels.csv + SEVERE_CLASSES).
DEFAULT_SEVERITY_WEIGHTS = {1: 10, 2: 5, 3: 2, 4: 1}
SEVERE_CLASSES = (1, 2)  # classes treated as "severe" for the binary severe-vs-not view

TIME_BLOCKS = [
    "00-04 (late night)",
    "04-08 (early morning)",
    "08-12 (morning)",
    "12-16 (afternoon)",
    "16-20 (evening)",
    "20-24 (night)",
]
NIGHT_BLOCKS = {TIME_BLOCKS[0], TIME_BLOCKS[5]}
EARLY_BLOCK = TIME_BLOCKS[1]

MONTH_NAMES = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
               7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
DAY_NAMES = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}

FACTOR_TITLES = {
    "cause": "Cause",
    "weather": "Weather",
    "road_condition": "Road condition",
    "road_feature": "Road feature",
    "veh1": "Vehicle 1",
    "veh2": "Vehicle 2",
    "dow": "Day of week",
    "month": "Month",
    "time_block": "Time of day",
    "severity": "Severity",
    "location_type": "Location type",
    "side": "Road side",
}


def load_labels() -> dict:
    """Return {column: {code: label}} merging built-in names with data/labels.csv."""
    labels = {"dow": dict(DAY_NAMES), "month": dict(MONTH_NAMES), "veh2": {0: "None"}}
    if LABELS_CSV.exists():
        lab = pd.read_csv(LABELS_CSV, comment="#")
        for _, r in lab.iterrows():
            labels.setdefault(str(r["column"]).strip(), {})[int(r["code"])] = str(r["label"]).strip()
    return labels


def label(labels: dict, col: str, code) -> str:
    """Human-readable name for a coded value."""
    if col == "time_block":
        return str(code)
    try:
        code_i = int(code)
    except (TypeError, ValueError):
        return str(code)
    return labels.get(col, {}).get(code_i, f"Code {code_i}")
