"""
Estimarea Pretului pentru Procesoare
CPU Price Prediction — Ridge, Random Forest, Gradient Boosting, XGBoost
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

plt.style.use("ggplot")
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)

import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "CPU_benchmark_v4.csv")
RATA_RON_USD = 4.39
RANDOM_STATE = 33550336
TEST_SIZE = 0.2

FEATURE_COLS = [
    'cpuMark', 'threadMark', 'TDP',
    'perf_per_core', 'perf_per_watt',
    'core_rank', 'log_cores',
    'series_encoded',
    'cat_Desktop', 'cat_Laptop', 'cat_Mobile/Embedded', 'cat_Server', 'cat_Unknown',
    'testDate'
]


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def extract_series(name):
    name = str(name); l = name.lower()
    if 'ryzen' in l:
        if 'threadripper' in l: return 'Threadripper'
        for s in ['9','7','5','3']:
            if f'ryzen {s}' in l or f'ryzen\u2122 {s}' in l: return f'Ryzen_{s}'
        return 'Ryzen'
    for s in ['i9','i7','i5','i3']:
        if s in l: return s.upper()
    if 'xeon' in l:
        if 'gold' in l: return 'Xeon_Gold'
        if 'platinum' in l: return 'Xeon_Platinum'
        if 'silver' in l: return 'Xeon_Silver'
        return 'Xeon'
    if 'epyc' in l: return 'EPYC'
    if 'pentium' in l: return 'Pentium'
    if 'celeron' in l: return 'Celeron'
    if 'athlon' in l: return 'Athlon'
    if 'apple' in l or 'm1' in l or 'm2' in l: return 'Apple'
    if 'hygon' in l: return 'Hygon'
    return 'Other'


def extract_gen(name):
    name = str(name)
    m = re.search(r'(?:i[3-9]-)?(\d{4,5})[A-Za-z]*', name)
    if m:
        num = int(m.group(1)); s = str(num)
        if len(s) == 5: return int(s[:2])
        elif len(s) == 4: return int(s[:2]) if s[0] == '1' else int(s[0])
    return -1


def extract_tier(name):
    name = str(name); l = name.lower()
    if 'i9' in l or 'ryzen 9' in l or 'threadripper' in l: return 9
    if 'i7' in l or 'ryzen 7' in l: return 7
    if 'xeon platinum' in l: return 9
    if 'i5' in l or 'ryzen 5' in l or 'xeon gold' in l: return 5
    if 'i3' in l or 'ryzen 3' in l: return 3
    if 'epyc' in l: return 8
    if 'xeon' in l: return 4
    if 'pentium' in l: return 2
    if 'athlon' in l or 'celeron' in l: return 1
    return 0


def bin_cores(c):
    if c <= 4: return 0
    elif c <= 8: return 1
    elif c <= 16: return 2
    else: return 3


# ---------------------------------------------------------------------------
# Data loading & cleaning
# ---------------------------------------------------------------------------

def load_and_clean(path):
    df = pd.read_csv(path)
    df['manufacturer'] = df['cpuName'].apply(
        lambda x: str(x).split()[0] if pd.notnull(x) else 'Unknown')
    df['price'] = df['price'] * RATA_RON_USD
    df['log_price'] = np.log1p(df['price'])
    df['series'] = df['cpuName'].apply(extract_series)
    df['gen'] = df['cpuName'].apply(extract_gen)
    df['tier'] = df['cpuName'].apply(extract_tier)

    cols = ['manufacturer', 'series', 'gen', 'tier', 'log_price', 'price',
            'cpuMark', 'threadMark', 'TDP', 'cores', 'testDate', 'category']
    return df[cols].copy()


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df):
    fe = df.copy()
    fe['perf_per_core'] = fe['cpuMark'] / fe['cores']
    fe['perf_per_watt'] = fe['cpuMark'] / fe['TDP'].replace(0, np.nan)
    fe['core_rank'] = fe['cores'].apply(bin_cores)
    fe['log_cores'] = np.log1p(fe['cores'])

    # Elimin outlieri de preț extremi (< 50 RON)
    fe = fe[fe['price'] >= 50].copy()

    le_series = LabelEncoder()
    fe['series_encoded'] = le_series.fit_transform(fe['series'])

    # One-Hot Encoding pentru categorie
    fe['cat_simple'] = fe['category'].str.split(',').str[0].str.strip()
    cat_dummies = pd.get_dummies(fe['cat_simple'], prefix='cat')
    fe = pd.concat([fe, cat_dummies], axis=1)

    return fe.drop(columns=['category', 'cat_simple'], errors='ignore')


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def prepare_data(fe):
    model_data = fe.dropna()
    X = model_data[FEATURE_COLS]
    y = model_data['log_price']
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(name, y_true, y_pred, y_train=None, y_pred_train=None):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(np.expm1(y_true), np.expm1(y_pred))
    rmse = np.sqrt(mean_squared_error(np.expm1(y_true), np.expm1(y_pred)))
    mape = np.mean(
        np.abs((np.expm1(y_true) - np.expm1(y_pred)) / np.expm1(y_true))
    ) * 100

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  R² test:  {r2:.4f}")
    print(f"  MAE:      {mae:.2f} RON")
    print(f"  RMSE:     {rmse:.2f} RON")
    print(f"  MAPE:     {mape:.2f}%")

    if y_train is not None and y_pred_train is not None:
        r2_train = r2_score(y_train, y_pred_train)
        print(f"  R² train: {r2_train:.4f}")

    return {'name': name, 'r2': round(r2, 4), 'mae': round(mae, 2),
            'rmse': round(rmse, 2), 'mape': round(mape, 2)}


def plot_predicted_vs_actual(y_true, y_pred, title, ax):
    ax.scatter(y_true, y_pred, alpha=0.4, s=15, color='#1976D2')
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=2)
    ax.set_xlabel('Actual log_price')
    ax.set_ylabel('Predicted log_price')
    ax.set_title(title)


def plot_residuals(y_true, y_pred, ax):
    resid = y_true - y_pred
    ax.scatter(y_pred, resid, alpha=0.4, s=15, color='#D32F2F')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Predicted log_price')
    ax.set_ylabel('Residual')
    ax.set_title('Residual Plot')


def plot_feature_importance(importances, title, colors_map='viridis', ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.get_cmap(colors_map)(np.linspace(0.2, 0.8, len(importances)))
    importances.plot(kind='barh', color=colors, ax=ax)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Importanță')
    ax.grid(axis='x', linestyle='--', alpha=0.7)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def train_ridge(X_train_s, y_train, X_test_s, y_test, do_gridsearch=False):
    if do_gridsearch:
        param_grid = {'alpha': np.linspace(0, 100, 2000)}
        gs = GridSearchCV(Ridge(random_state=RANDOM_STATE), param_grid,
                          cv=5, scoring='r2', n_jobs=-1)
        gs.fit(X_train_s, y_train)
        model = gs.best_estimator_
        print(f"  Best alpha: {gs.best_params_['alpha']}")
        print(f"  Best CV R²: {gs.best_score_:.4f}")
    else:
        model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
        model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    return model, y_pred


def train_random_forest(X_train, y_train, X_test, y_test, do_gridsearch=False):
    if do_gridsearch:
        param_grid = {
            'n_estimators': [200, 500],
            'max_depth': [10, 20, 30, 40, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ["sqrt", "log2", 0.5, 0.8],
        }
        gs = GridSearchCV(RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
                          param_grid, cv=5, scoring='r2', n_jobs=-1)
        gs.fit(X_train, y_train)
        model = gs.best_estimator_
        print(f"  Best params: {gs.best_params_}")
        print(f"  Best CV R²: {gs.best_score_:.4f}")
    else:
        model = RandomForestRegressor(
            n_estimators=200, max_depth=15,
            min_samples_split=10, min_samples_leaf=4,
            random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred


def train_gradient_boost(X_train, y_train, X_test, y_test, do_gridsearch=False):
    if do_gridsearch:
        param_grid = {
            'n_estimators': [200, 500, 1000, 1500],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 8, 10],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'max_features': ['sqrt', 'log2', 0.8]
        }
        gs = RandomizedSearchCV(
            GradientBoostingRegressor(random_state=RANDOM_STATE),
            param_distributions=param_grid, n_iter=60,
            cv=5, scoring='r2', n_jobs=-1, random_state=RANDOM_STATE)
        gs.fit(X_train, y_train)
        model = gs.best_estimator_
        print(f"  Best params: {gs.best_params_}")
        print(f"  Best CV R²: {gs.best_score_:.4f}")
    else:
        model = GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            min_samples_split=10, min_samples_leaf=4,
            random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred


def train_xgboost(X_train, y_train, X_test, y_test, do_gridsearch=False):
    if do_gridsearch:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 7],
            'learning_rate': [0.03, 0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        gs = GridSearchCV(
            xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=0),
            param_grid, cv=5, scoring='r2', n_jobs=-1)
        gs.fit(X_train, y_train)
        model = gs.best_estimator_
        print(f"  Best params: {gs.best_params_}")
        print(f"  Best CV R²: {gs.best_score_:.4f}")
    else:
        model = xgb.XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
        model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred


# ---------------------------------------------------------------------------
# Feature importance plots
# ---------------------------------------------------------------------------

def plot_all_feature_importances(models_data):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for (name, importances, cmap), ax in zip(models_data, axes.flatten()):
        plot_feature_importance(importances, f'{name}', cmap, ax)
    plt.tight_layout()
    plt.show()


def plot_all_evaluations(y_test, preds_dict):
    n = len(preds_dict)
    fig, axes = plt.subplots(n, 2, figsize=(14, 5 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    for i, (name, y_pred) in enumerate(preds_dict.items()):
        r2 = r2_score(y_test, y_pred)
        plot_predicted_vs_actual(y_test, y_pred, f'{name} — R² = {r2:.4f}', axes[i, 0])
        plot_residuals(y_test, y_pred, axes[i, 1])
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  CPU Price Prediction")
    print("=" * 60)

    # 1. Load & clean
    df = load_and_clean(DATA_PATH)
    print(f"\nDate încărcate: {len(df)} rânduri")
    print(f"  Cu preț: {df['price'].notna().sum()}")
    print(f"  Fără preț: {df['price'].isna().sum()}")

    # 2. Feature engineering
    df_fe = engineer_features(df)
    print(f"\nDupă feature engineering: {len(df_fe)} rânduri")
    print(f"  (eliminați cei cu preț < 50 RON)")

    # 3. Train / test split
    X_train, X_test, y_train, y_test = prepare_data(df_fe)
    print(f"\nSplit: {len(X_train)} train, {len(X_test)} test")

    # Scale features for Ridge
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # -----------------------------------------------------------------------
    # 4. Feature importance (baseline models)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  FEATURE IMPORTANCE")
    print("=" * 60)

    ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    ridge.fit(X_train_s, y_train)
    imp_ridge = pd.Series(ridge.coef_, index=FEATURE_COLS).sort_values(ascending=True)

    rf = RandomForestRegressor(n_estimators=200, max_depth=15,
                                min_samples_split=10, min_samples_leaf=4,
                                random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    imp_rf = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)

    gb = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1,
                                    min_samples_split=10, min_samples_leaf=4,
                                    random_state=RANDOM_STATE)
    gb.fit(X_train, y_train)
    imp_gb = pd.Series(gb.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)

    xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8,
                                   random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
    xgb_model.fit(X_train, y_train)
    imp_xgb = pd.Series(xgb_model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)

    plot_all_feature_importances([
        ('Ridge (coef)', imp_ridge, 'coolwarm'),
        ('Random Forest', imp_rf, 'viridis'),
        ('Gradient Boost', imp_gb, 'berlin_r'),
        ('XGBoost', imp_xgb, 'plasma'),
    ])

    # -----------------------------------------------------------------------
    # 5. Baseline evaluation
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  BASELINE EVALUATION")
    print("=" * 60)

    results = []
    preds = {}

    # Ridge
    _, yp = train_ridge(X_train_s, y_train, X_test_s, y_test, do_gridsearch=False)
    results.append(evaluate_model('Ridge', y_test, yp))
    preds['Ridge'] = yp

    # Random Forest
    _, yp = train_random_forest(X_train, y_train, X_test, y_test, do_gridsearch=False)
    results.append(evaluate_model('Random Forest', y_test, yp))
    preds['Random Forest'] = yp

    # Gradient Boosting
    _, yp = train_gradient_boost(X_train, y_train, X_test, y_test, do_gridsearch=False)
    results.append(evaluate_model('Gradient Boost', y_test, yp))
    preds['Gradient Boost'] = yp

    # XGBoost
    _, yp = train_xgboost(X_train, y_train, X_test, y_test, do_gridsearch=False)
    results.append(evaluate_model('XGBoost', y_test, yp))
    preds['XGBoost'] = yp

    # Ensemble
    yp_ens = (preds['Random Forest'] + preds['Gradient Boost'] + preds['XGBoost']) / 3
    results.append(evaluate_model('Ensemble (RF+GB+XGB)', y_test, yp_ens))
    preds['Ensemble'] = yp_ens

    # Plots
    plot_all_evaluations(y_test, preds)

    # Summary table
    print("\n" + "=" * 60)
    print("  BASELINE SUMMARY")
    print("=" * 60)
    summary = pd.DataFrame(results)
    print(summary.to_string(index=False))

    # -----------------------------------------------------------------------
    # 6. Hyperparameter optimization
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 60)
    print("  HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)

    opt_results = []

    # Ridge GridSearch
    print("\n--- Ridge GridSearch ---")
    _, yp = train_ridge(X_train_s, y_train, X_test_s, y_test, do_gridsearch=True)
    opt_results.append(evaluate_model('Ridge (optimizat)', y_test, yp))

    # RF GridSearch
    print("\n--- Random Forest GridSearch ---")
    _, yp = train_random_forest(X_train, y_train, X_test, y_test, do_gridsearch=True)
    opt_results.append(evaluate_model('Random Forest (optimizat)', y_test, yp))

    # GB RandomizedSearch
    print("\n--- Gradient Boost RandomizedSearch ---")
    _, yp = train_gradient_boost(X_train, y_train, X_test, y_test, do_gridsearch=True)
    opt_results.append(evaluate_model('Gradient Boost (optimizat)', y_test, yp))

    # XGBoost GridSearch
    print("\n--- XGBoost GridSearch ---")
    _, yp = train_xgboost(X_train, y_train, X_test, y_test, do_gridsearch=True)
    opt_results.append(evaluate_model('XGBoost (optimizat)', y_test, yp))

    # Summary
    print("\n\n" + "=" * 60)
    print("  OPTIMIZED RESULTS")
    print("=" * 60)
    opt_summary = pd.DataFrame(opt_results)
    print(opt_summary.to_string(index=False))

    # -----------------------------------------------------------------------
    # Final comparison
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 60)
    print("  FINAL COMPARISON — Baseline vs Optimized")
    print("=" * 60)
    final = []
    for r in results:
        final.append({'Model': r['name'], 'Tip': 'Baseline',
                      'R²': r['r2'], 'MAE': r['mae'], 'MAPE': r['mape']})
    for r in opt_results:
        final.append({'Model': r['name'], 'Tip': 'Optimizat',
                      'R²': r['r2'], 'MAE': r['mae'], 'MAPE': r['mape']})
    final_df = pd.DataFrame(final)
    print(final_df.to_string(index=False))


if __name__ == '__main__':
    main()
