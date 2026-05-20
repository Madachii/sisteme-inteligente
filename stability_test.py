"""Stability test — different MIN_YEAR values"""
import pandas as pd, numpy as np, re, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

RATA_RON_USD = 4.39
RANDOM_STATE = 33550336
TEST_SIZE = 0.2
MAX_PRICE = 4000
MIN_PRICE = 50


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
    return 0


def is_consumer(name):
    kw = ['i3','i5','i7','i9','ryzen 3','ryzen 5','ryzen 7','ryzen 9','pentium','celeron','athlon']
    return any(k in str(name).lower() for k in kw)


def group_socket(sock):
    s = str(sock).lower()
    if 'lga' in s: return 'LGA'
    if 'am' in s: return 'AM'
    if 'bga' in s or 'pga' in s or 's1' in s: return 'Mobile'
    return 'Other'


def engineer_features(fe):
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

    le_series = LabelEncoder()
    fe['series_encoded'] = le_series.fit_transform(fe['series'])

    fe['cat_simple'] = fe['category'].str.split(',').str[0].str.strip()
    cat_dummies = pd.get_dummies(fe['cat_simple'], prefix='cat')
    for c in ['cat_Desktop', 'cat_Laptop']:
        if c not in cat_dummies.columns: cat_dummies[c] = 0
    fe = pd.concat([fe, cat_dummies], axis=1)

    fe['mfr_simple'] = fe['manufacturer'].apply(lambda x: x if x in ['Intel','AMD'] else 'Other')
    mfr_dummies = pd.get_dummies(fe['mfr_simple'], prefix='mfr')
    for c in ['mfr_AMD', 'mfr_Intel']:
        if c not in mfr_dummies.columns: mfr_dummies[c] = 0
    fe = pd.concat([fe, mfr_dummies], axis=1)

    fe['socket_group'] = fe['socket'].apply(group_socket)
    sock_dummies = pd.get_dummies(fe['socket_group'], prefix='sock')
    for c in ['sock_LGA','sock_AM','sock_Mobile']:
        if c not in sock_dummies.columns: sock_dummies[c] = 0
    fe = pd.concat([fe, sock_dummies], axis=1)

    return fe.drop(
        columns=['category','cat_simple','mfr_simple','manufacturer','socket_group','socket','testDate'],
        errors='ignore',
    )


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


def run_test(min_year, max_year, label):
    df = pd.read_csv('../data/CPU_benchmark_v4.csv')
    df['price_ron'] = df['price'] * RATA_RON_USD
    df['log_price'] = np.log1p(df['price_ron'])
    df['manufacturer'] = df['cpuName'].apply(lambda x: str(x).split()[0] if pd.notnull(x) else 'Unknown')
    df['series'] = df['cpuName'].apply(extract_series)
    df['gen'] = df['cpuName'].apply(extract_gen)
    df['tier'] = df['cpuName'].apply(extract_tier)

    df = df[df['price'].notna()].copy()
    df = df[(df['price_ron'] >= MIN_PRICE) & (df['price_ron'] <= MAX_PRICE)]
    df = df[df['category'].str.contains('Desktop|Laptop', na=False) & ~df['category'].str.contains('Server', na=False)]
    df = df[df['cpuName'].apply(is_consumer)]
    df = df[(df['testDate'] >= min_year) & (df['testDate'] <= max_year)]

    fe = engineer_features(df)
    md = fe.dropna()
    X = md[FEATURE_COLS]
    y = md['log_price']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    model = GradientBoostingRegressor(
        loss='huber', alpha=0.9, random_state=RANDOM_STATE,
        n_estimators=400, learning_rate=0.05, max_depth=5,
        subsample=0.8, max_features='sqrt',
        min_samples_split=8, min_samples_leaf=4,
    )
    model.fit(X_tr, y_tr)
    yp = model.predict(X_te)
    yp_tr = model.predict(X_tr)
    r2 = r2_score(y_te, yp)
    mae = mean_absolute_error(np.expm1(y_te), np.expm1(yp))
    r2_tr = r2_score(y_tr, yp_tr)

    print(f'  {label}: {len(md):3d} rows  train={len(X_tr):3d} test={len(X_te):3d}  '
          f'R²={r2:.4f}  MAE={mae:.0f} RON  TrainR²={r2_tr:.4f}')


print('Stability test — GB Huber across different MIN_YEAR:')
print()
for yr in [2014, 2015, 2016, 2017, 2018]:
    run_test(yr, 2022, f'>={yr}')

print()
print('Full range tests:')
run_test(2010, 2022, '2010-2022 (all)')
run_test(2016, 2020, '2016-2020')
run_test(2018, 2022, '2018-2022')
