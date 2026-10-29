"""Load and clean the accident CSV."""
import pandas as pd

from .config import TIME_BLOCKS

RENAME = {
    "Day_of_Week": "dow",
    "Accident_Location_A": "location_type",
    "Accident_Location_A_Chainage_km": "km",
    "Accident_Location_A_Chainage_km_RoadSide": "side",
    "Accident_Severity_C": "severity",
    "Causes_D": "cause",
    "Road_Feature_E": "road_feature",
    "Road_Condition_F": "road_condition",
    "Weather_Conditions_H": "weather",
    "Vehicle_Type_Involved_J_V1": "veh1",
    "Vehicle_Type_Involved_J_V2": "veh2",
}
REQUIRED = ["Date", "Time_of_Accident"] + list(RENAME)
CODE_COLS = ["dow", "location_type", "side", "severity", "cause", "road_feature",
             "road_condition", "weather", "veh1", "veh2"]


def time_block(hour: int) -> str:
    return TIME_BLOCKS[int(hour) // 4]


def clean(raw: pd.DataFrame, min_year: int = 2010):
    """Return (clean_df, report). Removes duplicates, bad dates/times and stray old records."""
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    report = {"rows_in_file": int(len(raw))}
    df = raw[REQUIRED].copy()

    before = len(df)
    df = df.drop_duplicates()
    report["exact_duplicates_removed"] = int(before - len(df))

    df = df.rename(columns=RENAME)
    df["date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True, errors="coerce")
    t = pd.to_datetime(df["Time_of_Accident"].astype(str).str.strip(), format="mixed", errors="coerce")
    df["hour"] = t.dt.hour

    bad = df["date"].isna() | df["hour"].isna() | df["km"].isna() | df["severity"].isna()
    report["unparseable_rows_removed"] = int(bad.sum())
    df = df[~bad]

    old = df["date"].dt.year < min_year
    report["pre_%d_rows_removed" % min_year] = int(old.sum())
    df = df[~old].drop(columns=["Date", "Time_of_Accident"])

    df["year"] = df["date"].dt.year.astype(int)
    df["month"] = df["date"].dt.month.astype(int)
    df["hour"] = df["hour"].astype(int)
    df["veh2"] = df["veh2"].fillna(0)
    for c in CODE_COLS:
        df[c] = df[c].fillna(0).astype(int)
    df["km"] = df["km"].astype(float)
    df["time_block"] = df["hour"].map(time_block)
    df["is_night"] = (df["hour"] >= 20) | (df["hour"] < 6)

    report["rows_used"] = int(len(df))
    return df.reset_index(drop=True), report
