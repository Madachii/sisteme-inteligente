"""
Test 10 moderne CPUs against both models (A: no val, B: with val)
Uses actual PassMark specs from May 2026 data
"""
import pandas as pd, numpy as np, re, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb, lightgbm as lgb

RATA_RON_USD = 4.39
RANDOM_STATE = 33550336
MAX_PRICE = 4000
MIN_PRICE = 50
MIN_YEAR = 2016

# ======== Helper functions (copy from optimized_model) =========
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
    if 'core ultra' in l or 'ultra' in l:
        for s in ['9','7','5']:
            if f'ultra {s}' in l: return f'I{s.upper()}'
        return 'Other'
    return 'Other'

def extract_gen(name):
    name = str(name); l = name.lower()
    if 'ryzen' in l:
        m = re.search(r'ryzen\s*\d\s*(\d)(\d{3})', l)
        if m: return int(m.group(1))
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
    if 'ultra 9' in l or 'core ultra 9' in l: return 9
    if 'ultra 7' in l or 'core ultra 7' in l: return 7
    if 'ultra 5' in l or 'core ultra 5' in l: return 5
    return 0

def is_consumer(name):
    kw = ['i3','i5','i7','i9','ryzen 3','ryzen 5','ryzen 7','ryzen 9','pentium','celeron','athlon','ultra 5','ultra 7','ultra 9','core ultra']
    return any(k in str(name).lower() for k in kw)

def group_socket(sock):
    s = str(sock).lower()
    if 'lga' in s: return 'LGA'
    if 'am' in s: return 'AM'
    if 'bga' in s or 'pga' in s or 's1' in s: return 'Mobile'
    return 'Other'

# ======== Load original data =========
df = pd.read_csv('../data/CPU_benchmark_v4.csv')
df['price_ron'] = df['price'] * RATA_RON_USD
df['log_price'] = np.log1p(df['price_ron'])
df['manufacturer'] = df['cpuName'].apply(lambda x: str(x).split()[0] if pd.notnull(x) else 'Unknown')
df['series'] = df['cpuName'].apply(extract_series)
df['gen'] = df['cpuName'].apply(extract_gen)
df['tier'] = df['cpuName'].apply(extract_tier)

# ======== Add new CPUs manually =========
new_cpus = [
    # name, price_usd_approx, cpuMark, threadMark, TDP, cores, testDate, socket, category
    # Prices commented out — we're predicting, not using
    ("AMD Ryzen 9 9950X3D2 Dual Ed.",   None, 88000, 4850, 350, 32, 2026, "AM5",             "Desktop"),
    ("Intel Core Ultra 7 270K Plus",    None, 68904, 5000, 125, 20, 2026, "FCLGA1851",       "Desktop"),
    ("Intel Core Ultra 5 250K Plus",    None, 51000, 4900, 125, 14, 2026, "FCLGA1851",       "Desktop"),
    ("AMD Ryzen 7 9850X3D",             None, 41388, 4500, 120,  8, 2026, "AM5",             "Desktop"),
    ("AMD Ryzen 9 9950X3D",             None, 70191, 4743, 170, 16, 2025, "AM5",             "Desktop"),
    ("AMD Ryzen 9 9900X3D",             None, 56210, 4600, 120, 12, 2025, "AM5",             "Desktop"),
    ("AMD Ryzen 7 9800X3D",             None, 39971, 4425, 120,  8, 2024, "AM5",             "Desktop"),
    ("Intel Core Ultra 9 285K",         None, 67286, 4950, 125, 24, 2024, "FCLGA1851",       "Desktop"),
    ("Intel Core Ultra 7 265K",         None, 58695, 4929, 125, 20, 2024, "FCLGA1851",       "Desktop"),
    ("Intel Core Ultra 5 245K",         None, 43153, 4800, 125, 14, 2024, "FCLGA1851",       "Desktop"),
]

new_rows = []
for name, price_usd, cm, tm, tdp, cores, td, sock, cat in new_cpus:
    new_rows.append({
        'cpuName': name,
        'price': price_usd,
        'price_ron': price_usd * RATA_RON_USD if price_usd else np.nan,
        'log_price': np.log1p(price_usd * RATA_RON_USD) if price_usd else np.nan,
        'manufacturer': str(name).split()[0],
        'series': extract_series(name),
        'gen': extract_gen(name),
        'tier': extract_tier(name),
        'cpuMark': cm,
        'threadMark': tm,
        'TDP': tdp,
        'cores': cores,
        'testDate': td,
        'socket': sock,
        'category': cat,
    })

new_df = pd.DataFrame(new_rows)
df = pd.concat([df, new_df], ignore_index=True)

# ======== Filter for training (original data only) =========
df_train = df[~df['cpuName'].isin([n for n,_,_,_,_,_,_,_,_ in new_cpus])].copy()
df_train = df_train[df_train['price'].notna()].copy()
df_train = df_train[(df_train['price_ron'] >= MIN_PRICE) & (df_train['price_ron'] <= MAX_PRICE)]
df_train = df_train[df_train['category'].str.contains('Desktop|Laptop', na=False) & ~df_train['category'].str.contains('Server', na=False)]
df_train = df_train[df_train['cpuName'].apply(is_consumer)]
df_train = df_train[df_train['testDate'] >= MIN_YEAR]

print(f'Training data: {len(df_train)} rows')

# ======== Feature engineering on training data =========
def engineer_all(fe):
    fe = fe.copy()
    fe['perf_per_core'] = fe['cpuMark'] / fe['cores']
    fe['tdp_missing'] = fe['TDP'].isna().astype(int)
    tdp_median = fe['TDP'].median()
    fe['TDP_imputed'] = fe['TDP'].fillna(tdp_median)
    fe['perf_per_watt'] = fe['cpuMark'] / fe['TDP_imputed'].replace(0, np.nan)
    fe['log_cores'] = np.log1p(fe['cores'])
    fe['log_cpuMark'] = np.log1p(fe['cpuMark'])
    fe['log_perf_per_core'] = np.log1p(fe['perf_per_core'])
    fe['age'] = 2025 - fe['testDate']
    fe['cpuMark_per_core'] = fe['log_cpuMark'] * fe['log_cores']
    return fe

fe_train = engineer_all(df_train)

le_series = LabelEncoder()
fe_train['series_encoded'] = le_series.fit_transform(fe_train['series'])

fe_train['cat_simple'] = fe_train['category'].str.split(',').str[0].str.strip()
cat_dummies = pd.get_dummies(fe_train['cat_simple'], prefix='cat')
for c in ['cat_Desktop', 'cat_Laptop']:
    if c not in cat_dummies.columns: cat_dummies[c] = 0
fe_train = pd.concat([fe_train, cat_dummies], axis=1)

fe_train['mfr_simple'] = fe_train['manufacturer'].apply(lambda x: x if x in ['Intel','AMD'] else 'Other')
mfr_dummies = pd.get_dummies(fe_train['mfr_simple'], prefix='mfr')
for c in ['mfr_AMD', 'mfr_Intel']:
    if c not in mfr_dummies.columns: mfr_dummies[c] = 0
fe_train = pd.concat([fe_train, mfr_dummies], axis=1)

fe_train['socket_group'] = fe_train['socket'].apply(group_socket)
sock_dummies = pd.get_dummies(fe_train['socket_group'], prefix='sock')
for c in ['sock_LGA','sock_AM','sock_Mobile']:
    if c not in sock_dummies.columns: sock_dummies[c] = 0
fe_train = pd.concat([fe_train, sock_dummies], axis=1)

FEATURE_COLS = [
    'cpuMark','threadMark','TDP_imputed','tdp_missing',
    'perf_per_core','perf_per_watt',
    'log_cores','log_cpuMark','log_perf_per_core',
    'cpuMark_per_core',
    'gen','tier','series_encoded','age',
    'cat_Desktop','cat_Laptop',
    'mfr_AMD','mfr_Intel',
    'sock_LGA','sock_AM','sock_Mobile',
]

md = fe_train.dropna()
X = md[FEATURE_COLS]
y = md['log_price']

# ======== Prepare test CPUs =========
df_test = df[df['cpuName'].isin([n for n,_,_,_,_,_,_,_,_ in new_cpus])].copy()
fe_test = engineer_all(df_test)

# Apply same encoders
fe_test['series_encoded'] = le_series.transform(fe_test['series'])

fe_test['cat_simple'] = fe_test['category'].str.split(',').str[0].str.strip()
ct_dummies = pd.get_dummies(fe_test['cat_simple'], prefix='cat')
for c in ['cat_Desktop', 'cat_Laptop']:
    if c not in ct_dummies.columns: ct_dummies[c] = 0
fe_test = pd.concat([fe_test, ct_dummies], axis=1)

fe_test['mfr_simple'] = fe_test['manufacturer'].apply(lambda x: x if x in ['Intel','AMD'] else 'Other')
mf_dummies = pd.get_dummies(fe_test['mfr_simple'], prefix='mfr')
for c in ['mfr_AMD', 'mfr_Intel']:
    if c not in mf_dummies.columns: mf_dummies[c] = 0
fe_test = pd.concat([fe_test, mf_dummies], axis=1)

fe_test['socket_group'] = fe_test['socket'].apply(group_socket)
sk_dummies = pd.get_dummies(fe_test['socket_group'], prefix='sock')
for c in ['sock_LGA','sock_AM','sock_Mobile']:
    if c not in sk_dummies.columns: sk_dummies[c] = 0
fe_test = pd.concat([fe_test, sk_dummies], axis=1)

X_new = fe_test[FEATURE_COLS]

# ======== Train Model A (80/20, no val) =========
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

gb_a = GradientBoostingRegressor(loss='huber',alpha=0.9,random_state=RANDOM_STATE,
    n_estimators=500,learning_rate=0.05,max_depth=4,subsample=0.6,max_features='sqrt',
    min_samples_split=20,min_samples_leaf=8)
gb_a.fit(X_tr, y_tr)
preds_a = np.expm1(gb_a.predict(X_new))

# ======== Train Model B (70/15/15, WITH val + early stopping) =========
X_temp, X_te_b, y_temp, y_te_b = train_test_split(X, y, test_size=0.15, random_state=RANDOM_STATE)
X_tr_b, X_val, y_tr_b, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=RANDOM_STATE)

gb_b = GradientBoostingRegressor(loss='huber',alpha=0.9,random_state=RANDOM_STATE,
    n_estimators=500,learning_rate=0.05,max_depth=4,subsample=0.6,max_features='sqrt',
    min_samples_split=20,min_samples_leaf=8)
gb_b.fit(X_tr_b, y_tr_b)
preds_b = np.expm1(gb_b.predict(X_new))

# XGB with early stopping
xgb_b = xgb.XGBRegressor(random_state=RANDOM_STATE,n_jobs=-1,verbosity=0,
    n_estimators=500,learning_rate=0.05,max_depth=4,subsample=0.6,colsample_bytree=0.6,
    reg_alpha=0.5,reg_lambda=1.0,min_child_weight=5,gamma=0.5,early_stopping_rounds=15)
xgb_b.fit(X_tr_b, y_tr_b, eval_set=[(X_val, y_val)], verbose=False)
preds_xgb_b = np.expm1(xgb_b.predict(X_new))

# LGB with early stopping
lgb_b = lgb.LGBMRegressor(objective='regression',random_state=RANDOM_STATE,n_jobs=-1,verbose=-1,
    n_estimators=500,learning_rate=0.05,max_depth=4,num_leaves=15,subsample=0.6,colsample_bytree=0.6,
    reg_alpha=0.5,reg_lambda=2.0,min_child_samples=20)
callbacks = [lgb.early_stopping(15), lgb.log_evaluation(0)]
lgb_b.fit(X_tr_b, y_tr_b, eval_set=[(X_val, y_val)], eval_metric='rmse', callbacks=callbacks)
preds_lgb_b = np.expm1(lgb_b.predict(X_new))

gb_r2_a = r2_score(y_te, gb_a.predict(X_te))
gb_mae_a = mean_absolute_error(np.expm1(y_te), np.expm1(gb_a.predict(X_te)))
gb_r2_b = r2_score(y_te_b, gb_b.predict(X_te_b))
gb_mae_b = mean_absolute_error(np.expm1(y_te_b), np.expm1(gb_b.predict(X_te_b)))

# ======== Results Table =========
print()
print("=" * 95)
print("  CPU PRICE PREDICTIONS — 10 MODERN CPUs")
print("=" * 95)
print(f"{'#':<2} {'CPU':<35s} {'Model A (GB)':>12s} {'Model B (GB)':>12s} {'Model B (XGB)':>13s} {'Model B (LGB)':>13s}")
print(f"{'':2} {'':35s} {'no val':>12s} {'with val':>12s} {'early_stop':>13s} {'early_stop':>13s}")
print("-" * 95)
cpu_names = [n for n,_,_,_,_,_,_,_,_ in new_cpus]
for i in range(len(cpu_names)):
    name = cpu_names[i][:34]
    pa = preds_a[i]
    pb = preds_b[i]
    px = preds_xgb_b[i]
    pl = preds_lgb_b[i]
    flag = " ⚠ $>" if pa > MAX_PRICE else ""
    print(f"{i+1:<2} {name:<35s} {pa:>8.0f} RON{flag}  {pb:>8.0f} RON{flag}  {px:>9.0f} RON{flag}  {pl:>9.0f} RON{flag}")

print("-" * 95)
print(f"\nModel A (80/20, no val): GB Huber  Test R²={gb_r2_a:.4f}  MAE={gb_mae_a:.0f} RON")
print(f"Model B (70/15/15, val): GB Huber  Test R²={gb_r2_b:.4f}  MAE={gb_mae_b:.0f} RON")
print(f"\n⚠ Prices above {MAX_PRICE} RON are outside model's training range (max {MAX_PRICE} RON)")
print("  Model extrapolating → predictions less reliable")
print()
print("Known actual prices (PassMark May 2026) for reference:")
print("  Ryzen 7 9800X3D:  $440 USD (~1930 RON)")
print("  Core Ultra 7 265K: $320 USD (~1405 RON)")
print("  Ryzen 9 9950X3D:  $640 USD (~2810 RON)")
print("  Core Ultra 5 245K: ~$250 USD (~1100 RON)")
print("  Core Ultra 9 285K: ~$450 USD (~1975 RON)")
