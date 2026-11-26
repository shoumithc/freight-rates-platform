"""Feature engineering shared by the training notebook and the Flask API."""
import numpy as np
import pandas as pd

SERIES_KEY = ["origin_locode", "dest_locode", "container_type"]
PRICE_WEEKS = [f"price_week_{k}" for k in range(1, 5)]   # price 1..4 weeks before the forecast week

US_COAST = {"USLAX": "West", "USLGB": "West", "USOAK": "West", "USSEA": "West", "USTIW": "West",
            "USNYC": "East", "USSAV": "East", "USCHS": "East", "USHOU": "Gulf"}

CATEGORICAL_FEATURES = ["origin_locode", "dest_locode", "container_type", "us_coast", "trade"]
NUMERICAL_FEATURES = ["last_week_vs_avg", "week4_vs_avg", "week_of_year",
                      "market_move_lag_1", "market_move_lag_2", "market_move_lag_3"]
FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
RAW_INPUTS = ["origin_locode", "dest_locode", "container_type", "rate_date"] + PRICE_WEEKS


def add_recent_prices(history: pd.DataFrame) -> pd.DataFrame:
    """Training only: add price_week_1..4, price_avg_4w and the target rate_move to weekly history."""
    df = history.sort_values(SERIES_KEY + ["rate_date"]).copy()
    grp = df.groupby(SERIES_KEY)["price"]
    for k in range(1, 5):
        df[f"price_week_{k}"] = grp.shift(k)
    df["price_avg_4w"] = df[PRICE_WEEKS].mean(axis=1)
    df["rate_move"] = df["price"] / df["price_avg_4w"]
    return df


def trade_of(origin: pd.Series, dest: pd.Series) -> pd.Series:
    """Trade = origin country - destination country, e.g. 'CN-US' (China -> US) or 'US-IN' (US -> India)."""
    return origin.astype(str).str.upper().str[:2] + "-" + dest.astype(str).str.upper().str[:2]


def build_market_moves(df: pd.DataFrame) -> pd.DataFrame:
    """Average rate_move of all lanes in the same trade, per week (the market index)."""
    tmp = df.dropna(subset=["rate_move"]).assign(trade=lambda d: trade_of(d["origin_locode"], d["dest_locode"]))
    market = (tmp.groupby(["trade", "rate_date"])["rate_move"].mean()
                 .rename("market_move").reset_index())
    market["rate_date"] = pd.to_datetime(market["rate_date"]).astype("datetime64[ns]")
    return market.sort_values(["trade", "rate_date"]).reset_index(drop=True)


def add_market_lags(df: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """market_move_lag_k = market index for the week k weeks before rate_date.
    If that week is not in the table yet, the latest earlier week is used."""
    out = df.copy()
    out["_row"] = np.arange(len(out))
    for k in (1, 2, 3):
        left = out[["_row", "trade", "rate_date"]].copy()
        left["_week"] = (left["rate_date"] - pd.Timedelta(weeks=k)).astype("datetime64[ns]")
        right = market.rename(columns={"rate_date": "_week", "market_move": f"market_move_lag_{k}"})
        merged = pd.merge_asof(left.sort_values("_week"), right.sort_values("_week"),
                               on="_week", by="trade", direction="backward")
        out = out.merge(merged[["_row", f"market_move_lag_{k}"]], on="_row", how="left")
    return out.sort_values("_row").drop(columns="_row").reset_index(drop=True)


def engineer_features(df: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Raw inputs (RAW_INPUTS) -> model features, plus price_avg_4w and market_anchor
    (= price_avg_4w x market_move_lag_1), which turns the model's correction back into a price."""
    missing = [c for c in RAW_INPUTS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required fields: {missing}")
    out = df.copy()
    out["origin_locode"] = out["origin_locode"].astype(str).str.upper().str.strip()
    out["dest_locode"] = out["dest_locode"].astype(str).str.upper().str.strip()
    out["container_type"] = out["container_type"].astype(str).str.upper().str.strip()
    out["rate_date"] = pd.to_datetime(out["rate_date"]).astype("datetime64[ns]")
    for c in PRICE_WEEKS:
        out[c] = pd.to_numeric(out[c], errors="raise").astype(float)
    if (out[PRICE_WEEKS] <= 0).any().any():
        raise ValueError("price_week_1..4 must all be greater than 0")

    out["price_avg_4w"] = out[PRICE_WEEKS].mean(axis=1)
    out["last_week_vs_avg"] = out["price_week_1"] / out["price_avg_4w"]
    out["week4_vs_avg"] = out["price_week_4"] / out["price_avg_4w"]
    out["week_of_year"] = out["rate_date"].dt.isocalendar().week.astype(int)
    us_end = np.where(out["origin_locode"].str.startswith("US"), out["origin_locode"], out["dest_locode"])
    out["us_coast"] = pd.Series(us_end, index=out.index).map(US_COAST).fillna("Other")
    out["trade"] = trade_of(out["origin_locode"], out["dest_locode"])
    out = add_market_lags(out, market)
    out["market_anchor"] = out["price_avg_4w"] * out["market_move_lag_1"]
    return out
