"""
CPU Price Prediction — Consumer-Only (no server pollution)
"""

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "CPU_benchmark_v4.csv")
RATA_RON_USD = 4.39
RANDOM_STATE = 33550336
TEST_SIZE = 0.2
MIN_YEAR = 2016
MAX_PRICE = 4000
MIN_PRICE = 50

def is_consumer(name):
    kw = [
        "i3", "i5", "i7", "i9",
        "ryzen 3", "ryzen 5", "ryzen 7", "ryzen 9",
        "pentium", "celeron", "athlon",
    ]
    return any(k in str(name).lower() for k in kw)


def extract_series(name):
    name = str(name)
    l = name.lower()
    if "ryzen" in l:
        if "threadripper" in l:
            return "Threadripper"
        for s in ["9", "7", "5", "3"]:
            if f"ryzen {s}" in l or f"ryzen\u2122 {s}" in l:
                return f"Ryzen_{s}"
        return "Ryzen"
    for s in ["i9", "i7", "i5", "i3"]:
        if s in l:
            return s.upper()
    if "xeon" in l:
        if "gold" in l:
            return "Xeon_Gold"
        if "platinum" in l:
            return "Xeon_Platinum"
        if "silver" in l:
            return "Xeon_Silver"
        return "Xeon"
    if "epyc" in l:
        return "EPYC"
    if "pentium" in l:
        return "Pentium"
    if "celeron" in l:
        return "Celeron"
    if "athlon" in l:
        return "Athlon"
    return "Other"


def extract_gen(name):
    name = str(name)
    l = name.lower()

    # AMD Ryzen: extract first digit after "ryzen X" (e.g., "Ryzen 5 5600X" → 5000 series → 5)
    if "ryzen" in l:
        m = re.search(r"ryzen\s*\d\s*(\d)(\d{3})", l)
        if m:
            return int(m.group(1))

    # Intel: match 4-5 digit model numbers
    m = re.search(r"(?:i[3-9]-)?(\d{4,5})[A-Za-z]*", name)
    if m:
        num = int(m.group(1))
        s = str(num)
        if len(s) == 5:
            return int(s[:2])
        elif len(s) == 4:
            return int(s[:2]) if s[0] == "1" else int(s[0])
    return -1


def extract_tier(name):
    name = str(name)
    l = name.lower()
    if "i9" in l or "ryzen 9" in l or "threadripper" in l:
        return 9
    if "i7" in l or "ryzen 7" in l:
        return 7
    if "xeon platinum" in l:
        return 9
    if "i5" in l or "ryzen 5" in l or "xeon gold" in l:
        return 5
    if "i3" in l or "ryzen 3" in l:
        return 3
    if "epyc" in l:
        return 8
    if "xeon" in l:
        return 4
    if "pentium" in l:
        return 2
    if "athlon" in l or "celeron" in l:
        return 1
    return 0


def load_and_filter(path):
    df = pd.read_csv(path)
    df["price_ron"] = df["price"] * RATA_RON_USD
    df["log_price"] = np.log1p(df["price_ron"])
    df["manufacturer"] = df["cpuName"].apply(
        lambda x: str(x).split()[0] if pd.notnull(x) else "Unknown"
    )
    df["series"] = df["cpuName"].apply(extract_series)
    df["gen"] = df["cpuName"].apply(extract_gen)
    df["tier"] = df["cpuName"].apply(extract_tier)

    df = df[df["price"].notna()].copy()
    df = df[df["price_ron"] >= MIN_PRICE]
    df = df[df["price_ron"] <= MAX_PRICE]

    # Consumer-only: Desktop/Laptop, no Server, known consumer series
    df = df[df["category"].str.contains("Desktop|Laptop", na=False)]
    df = df[~df["category"].str.contains("Server", na=False)]
    df = df[df["cpuName"].apply(is_consumer)]

    # Filter old CPUs (PassMark zombie prices)
    df = df[df["testDate"] >= MIN_YEAR]

    cols = [
        "manufacturer", "series", "gen", "tier",
        "log_price", "price_ron",
        "cpuMark", "threadMark", "TDP", "cores",
        "testDate", "socket", "category",
    ]
    return df[cols].copy()


def engineer_features(df):
    fe = df.copy()

    fe["perf_per_core"] = fe["cpuMark"] / fe["cores"]

    fe["tdp_missing"] = fe["TDP"].isna().astype(int)
    tdp_median = fe["TDP"].median()
    fe["TDP_imputed"] = fe["TDP"].fillna(tdp_median)
    fe["perf_per_watt"] = fe["cpuMark"] / fe["TDP_imputed"].replace(0, np.nan)

    fe["log_cores"] = np.log1p(fe["cores"])
    fe["log_cpuMark"] = np.log1p(fe["cpuMark"])
    fe["log_perf_per_core"] = np.log1p(fe["perf_per_core"])

    # Age feature: how old the CPU is
    fe["age"] = 2025 - fe["testDate"]

    # Interaction
    fe["cpuMark_per_core"] = fe["log_cpuMark"] * fe["log_cores"]

    # Encode series
    le_series = LabelEncoder()
    fe["series_encoded"] = le_series.fit_transform(fe["series"])

    # One-hot category
    fe["cat_simple"] = fe["category"].str.split(",").str[0].str.strip()
    cat_dummies = pd.get_dummies(fe["cat_simple"], prefix="cat")
    for c in ["cat_Desktop", "cat_Laptop"]:
        if c not in cat_dummies.columns:
            cat_dummies[c] = 0
    fe = pd.concat([fe, cat_dummies], axis=1)

    # One-hot manufacturer (Intel vs AMD is key)
    fe["mfr_simple"] = fe["manufacturer"].apply(
        lambda x: x if x in ["Intel", "AMD"] else "Other"
    )
    mfr_dummies = pd.get_dummies(fe["mfr_simple"], prefix="mfr")
    for c in ["mfr_AMD", "mfr_Intel"]:
        if c not in mfr_dummies.columns:
            mfr_dummies[c] = 0
    fe = pd.concat([fe, mfr_dummies], axis=1)

    # Socket grouping
    def group_socket(sock):
        s = str(sock).lower()
        if "lga" in s:
            return "LGA"
        if "am" in s:
            return "AM"
        if "bga" in s or "pga" in s or "s1" in s:
            return "Mobile"
        return "Other"

    fe["socket_group"] = fe["socket"].apply(group_socket)
    sock_dummies = pd.get_dummies(fe["socket_group"], prefix="sock")
    for c in ["sock_LGA", "sock_AM", "sock_Mobile"]:
        if c not in sock_dummies.columns:
            sock_dummies[c] = 0
    fe = pd.concat([fe, sock_dummies], axis=1)

    return fe.drop(
        columns=["category", "cat_simple", "mfr_simple", "manufacturer",
                 "socket_group", "socket", "testDate"],
        errors="ignore",
    )


FEATURE_COLS = [
    "cpuMark", "threadMark", "TDP_imputed", "tdp_missing",
    "perf_per_core", "perf_per_watt",
    "log_cores", "log_cpuMark", "log_perf_per_core",
    "cpuMark_per_core",
    "gen", "tier", "series_encoded", "age",
    "cat_Desktop", "cat_Laptop",
    "mfr_AMD", "mfr_Intel",
    "sock_LGA", "sock_AM", "sock_Mobile",
]


def prepare_data(fe):
    model_data = fe.dropna()
    X = model_data[FEATURE_COLS]
    y = model_data["log_price"]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, random_state=RANDOM_STATE
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_model(name, y_true, y_pred, y_train=None, y_pred_train=None):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(np.expm1(y_true), np.expm1(y_pred))
    rmse = np.sqrt(mean_squared_error(np.expm1(y_true), np.expm1(y_pred)))
    mape = np.mean(
        np.abs(
            (np.expm1(y_true) - np.expm1(y_pred)) / np.expm1(y_true)
        )
    ) * 100

    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    print(f"  R² test:  {r2:.4f}")
    print(f"  MAE:      {mae:.2f} RON")
    print(f"  RMSE:     {rmse:.2f} RON")
    print(f"  MAPE:     {mape:.2f}%")

    if y_train is not None and y_pred_train is not None:
        r2_train = r2_score(y_train, y_pred_train)
        print(f"  R² train: {r2_train:.4f}")

    return {
        "name": name, "r2": round(r2, 4),
        "mae": round(mae, 2), "rmse": round(rmse, 2),
        "mape": round(mape, 2),
    }


# ---------------------------------------------------------------------------
# Models — lighter hyperparameter search
# ---------------------------------------------------------------------------

def train_gb(X_train, X_val, y_train, y_val, X_test, y_test):
    model = GradientBoostingRegressor(
        loss="huber", alpha=0.9, random_state=RANDOM_STATE,
        n_estimators=500, learning_rate=0.05, max_depth=4,
        subsample=0.6, max_features="sqrt",
        min_samples_split=20, min_samples_leaf=8,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    return model, y_pred, y_pred_train


def train_xgb(X_train, X_val, y_train, y_val, X_test, y_test):
    model = xgb.XGBRegressor(
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
        n_estimators=500, learning_rate=0.05, max_depth=4,
        subsample=0.6, colsample_bytree=0.6,
        reg_alpha=0.5, reg_lambda=1.0,
        min_child_weight=5, gamma=0.5,
        early_stopping_rounds=15,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    return model, y_pred, y_pred_train


def train_rf(X_train, X_val, y_train, y_val, X_test, y_test):
    model = RandomForestRegressor(
        n_estimators=300, max_depth=12,
        min_samples_split=15, min_samples_leaf=5,
        max_features="sqrt",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    return model, y_pred, y_pred_train


def train_lgb(X_train, X_val, y_train, y_val, X_test, y_test):
    callbacks = None
    try:
        import lightgbm as _lgb
        if hasattr(_lgb, 'early_stopping'):
            callbacks = [_lgb.early_stopping(15), _lgb.log_evaluation(0)]
    except:
        pass

    model = lgb.LGBMRegressor(
        objective="regression",
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        n_estimators=500, learning_rate=0.05, max_depth=4,
        num_leaves=15, subsample=0.6, colsample_bytree=0.6,
        reg_alpha=0.5, reg_lambda=2.0,
        min_child_samples=20,
    )
    fit_kw = dict(eval_set=[(X_val, y_val)], eval_metric="rmse")
    if callbacks:
        fit_kw["callbacks"] = callbacks
    model.fit(X_train, y_train, **fit_kw)
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    return model, y_pred, y_pred_train


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  CONSUMER CPU Price Prediction")
    print("=" * 60)

    df = load_and_filter(DATA_PATH)
    print(f"\nConsumer data: {len(df)} rows")
    print(f"  Year range: {df['testDate'].min():.0f} - {df['testDate'].max():.0f}")
    print(f"  Price range: {df['price_ron'].min():.0f} - {df['price_ron'].max():.0f} RON")
    print(f"  Median price: {df['price_ron'].median():.0f} RON")

    df_fe = engineer_features(df)
    print(f"After feature engineering: {len(df_fe)} rows, {len(FEATURE_COLS)} features")

    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(df_fe)
    print(f"Split: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test\n")

    results = []
    preds = {}

    # Gradient Boosting (Huber)
    print("--- Gradient Boosting (Huber) ---")
    gb_model, yp_gb, yp_gb_train = train_gb(X_train, X_val, y_train, y_val, X_test, y_test)
    results.append(evaluate_model("GB Huber", y_test, yp_gb, y_train, yp_gb_train))
    preds["GB"] = yp_gb

    # XGBoost
    print("--- XGBoost ---")
    xgb_model, yp_xgb, yp_xgb_train = train_xgb(X_train, X_val, y_train, y_val, X_test, y_test)
    results.append(evaluate_model("XGBoost", y_test, yp_xgb, y_train, yp_xgb_train))
    preds["XGB"] = yp_xgb

    # Random Forest
    print("--- Random Forest ---")
    rf_model, yp_rf, yp_rf_train = train_rf(X_train, X_val, y_train, y_val, X_test, y_test)
    results.append(evaluate_model("Random Forest", y_test, yp_rf, y_train, yp_rf_train))
    preds["RF"] = yp_rf

    # LightGBM
    print("--- LightGBM ---")
    lgb_model, yp_lgb, yp_lgb_train = train_lgb(X_train, X_val, y_train, y_val, X_test, y_test)
    results.append(evaluate_model("LightGBM", y_test, yp_lgb, y_train, yp_lgb_train))
    preds["LGB"] = yp_lgb

    # Ridge baseline
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    ridge = RidgeCV(alphas=np.logspace(-2, 3, 50), cv=5)
    ridge.fit(X_train_s, y_train)
    yp_ridge = ridge.predict(X_test_s)
    yp_ridge_train = ridge.predict(X_train_s)
    results.append(evaluate_model("Ridge CV", y_test, yp_ridge, y_train, yp_ridge_train))

    # Ensembles
    yp_avg = (yp_gb + yp_xgb + yp_rf + yp_lgb) / 4
    results.append(evaluate_model("Avg Ensemble", y_test, yp_avg))

    yp_wavg = 0.30 * yp_rf + 0.25 * yp_gb + 0.25 * yp_lgb + 0.20 * yp_xgb
    results.append(evaluate_model("Weighted Ensemble", y_test, yp_wavg))

    # Summary
    print("\n" + "=" * 60)
    print("  RESULTS — Consumer CPUs")
    print("=" * 60)
    summary = pd.DataFrame(results)
    print(summary.sort_values("r2", ascending=False).to_string(index=False))

    best = summary.loc[summary["mae"].idxmin()]
    print(f"\nBest model: {best['name']}")
    print(f"  R² = {best['r2']:.4f}")
    print(f"  MAE = {best['mae']:.2f} RON")
    print(f"  MAPE = {best['mape']:.2f}%")


if __name__ == "__main__":
    main()
