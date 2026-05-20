"""
ml_model.py — MediSight ML Engine v4
12 models: Admission Forecast, Disease Risk, Anomaly Detection,
Patient Clustering, Bed Demand, Doctor Workload, Outbreak Warning,
Readmission Prediction, Survival Analysis, Cost Prediction,
Monte Carlo Simulation, Symptom Classifier
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, f1_score, roc_auc_score, precision_score, recall_score

try:
    from prophet import Prophet
    PROPHET_OK = True
except Exception:
    PROPHET_OK = False

try:
    import shap
    SHAP_OK = True
except Exception:
    SHAP_OK = False

try:
    import xgboost as xgb
    XGB_OK = True
except Exception:
    XGB_OK = False

try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    LIFELINES_OK = True
except Exception:
    LIFELINES_OK = False

try:
    from scipy.integrate import odeint
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# ══════════════════════════════════════════════════════════
#  SHARED HELPERS
# ══════════════════════════════════════════════════════════

def _daily_series(df, group_col=None, group_val=None):
    sub = df.dropna(subset=['Admission_Date'])
    if group_col and group_val:
        sub = sub[sub[group_col] == group_val]
    return (
        sub.groupby(sub['Admission_Date'].dt.date)
        .size()
        .reset_index(name='count')
        .rename(columns={'Admission_Date': 'date'})
        .sort_values('date')
        .reset_index(drop=True)
    )


def _prophet_forecast(daily, forecast_days):
    if not PROPHET_OK or len(daily) < 14:
        return None, None
    try:
        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(daily['date']),
            'y':  daily['count'].astype(float),
        })
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            interval_width=0.80,
        )
        m.fit(prophet_df)
        future = m.make_future_dataframe(periods=forecast_days)
        fc = m.predict(future)
        fc_future = fc.tail(forecast_days)
        forecast_list = [
            {
                'date':  str(row['ds'].date()),
                'count': max(0, round(float(row['yhat']))),
                'lower': max(0, round(float(row['yhat_lower']))),
                'upper': max(0, round(float(row['yhat_upper']))),
                'type':  'predicted',
            }
            for _, row in fc_future.iterrows()
        ]
        return forecast_list, m
    except Exception:
        return None, None


def _build_temporal_features(dates_series, day_idx_series):
    """
    Build a rich feature matrix from date information.
    Features: day_idx (trend), day_of_week, month, week_of_year,
              is_weekend, quarter, plus Fourier terms for weekly
              and yearly seasonality.
    Forces pd.DatetimeIndex so .dayofweek / .isocalendar() work
    whether dates_series is a Series of date objects or a DatetimeIndex.
    """
    dt  = pd.DatetimeIndex(pd.to_datetime(list(dates_series)))
    idx = np.array(day_idx_series, dtype=float)
    return np.column_stack([
        idx,
        dt.dayofweek.astype(float),
        dt.month.astype(float),
        np.array(dt.isocalendar().week, dtype=float),
        (dt.dayofweek >= 5).astype(float),
        dt.quarter.astype(float),
        np.sin(2 * np.pi * idx / 7),
        np.cos(2 * np.pi * idx / 7),
        np.sin(2 * np.pi * idx / 365.25),
        np.cos(2 * np.pi * idx / 365.25),
    ])


def _build_lag_features(counts_array):
    """
    Lag / rolling features from a 1-D count array.
    Returns columns: lag1, lag7, lag14, roll3, roll7
    bfill ensures no NaNs in early rows.
    """
    s     = pd.Series(counts_array.astype(float))
    lag1  = s.shift(1).bfill().values
    lag7  = s.shift(7).bfill().values
    lag14 = s.shift(14).bfill().values
    roll3 = s.rolling(3,  min_periods=1).mean().shift(1).bfill().values
    roll7 = s.rolling(7,  min_periods=1).mean().shift(1).bfill().values
    return np.column_stack([lag1, lag7, lag14, roll3, roll7])


def _lr_forecast(daily, forecast_days, smooth=True):
    if len(daily) < 5:
        return None, None, None
    daily = daily.copy()
    if smooth:
        daily['count'] = (
            daily['count']
            .rolling(window=7, min_periods=3, center=False)
            .mean()
            .fillna(daily['count'])
            .round()
            .astype(int)
        )
    daily['day_idx'] = range(len(daily))

    y   = daily['count'].values.astype(float)
    X_t = _build_temporal_features(daily['date'], daily['day_idx'])
    X_l = _build_lag_features(y)
    # Train on temporal + lag features — dramatically raises R²
    X   = np.hstack([X_t, X_l])
    m   = LinearRegression().fit(X, y)

    # ── Iterative forecasting ──────────────────────────────────────────
    # Future lags are unknown, so we forecast one day at a time,
    # feeding each prediction back into the lag buffer.
    last_date    = pd.to_datetime(daily['date'].max())
    last_idx     = int(daily['day_idx'].max())
    count_buffer = list(y)          # grows as we predict each day

    forecast = []
    for i in range(1, forecast_days + 1):
        fut_date = last_date + timedelta(days=i)
        fut_idx  = last_idx + i
        X_t_row  = _build_temporal_features(
            pd.DatetimeIndex([fut_date]), [fut_idx]
        )                                                   # (1, 10)

        buf = np.array(count_buffer, dtype=float)
        l1  = buf[-1]
        l7  = buf[-7]  if len(buf) >= 7  else buf[0]
        l14 = buf[-14] if len(buf) >= 14 else buf[0]
        r3  = buf[-3:].mean() if len(buf) >= 3 else buf.mean()
        r7  = buf[-7:].mean() if len(buf) >= 7 else buf.mean()
        X_l_row  = np.array([[l1, l7, l14, r3, r7]])       # (1, 5)
        X_row    = np.hstack([X_t_row, X_l_row])            # (1, 15)

        pred_val = max(0.0, float(m.predict(X_row)[0]))
        count_buffer.append(round(pred_val))

        forecast.append({
            'date':  str(fut_date.date()),
            'count': max(0, round(pred_val)),
            'lower': None,
            'upper': None,
            'type':  'predicted',
        })

    # Return X alongside so train_and_forecast can score without rebuilding
    return m, forecast, daily, X


# ══════════════════════════════════════════════════════════
#  MODEL 1 — train_and_forecast
# ══════════════════════════════════════════════════════════

def train_and_forecast(df, forecast_days=14):
    daily_raw = _daily_series(df)
    actual = [
        {'date': str(r['date']), 'count': int(r['count']), 'type': 'actual'}
        for _, r in daily_raw.iterrows()
    ]
    if len(daily_raw) < 7:
        return {
            'actual': actual, 'predicted': [],
            'model': 'Linear Regression', 'model_r2': 0,
            'has_ci': False, 'engine': 'linear_regression',
        }

    fc_list, _ = _prophet_forecast(daily_raw, forecast_days)
    if fc_list is not None:
        return {
            'actual': actual, 'predicted': fc_list,
            'model': 'Prophet (Seasonality + Trend + CI)',
            'model_r2': None, 'has_ci': True, 'engine': 'prophet',
        }

    m, forecast, daily_smooth, X = _lr_forecast(daily_raw, forecast_days, smooth=True)
    if m is None:
        return {'actual': actual, 'predicted': [], 'model': 'Linear Regression',
                'model_r2': 0, 'has_ci': False, 'engine': 'linear_regression'}
    # X already contains temporal + lag features used during training
    y = daily_smooth['count'].values.astype(float)
    return {
        'actual': actual, 'predicted': forecast,
        'model': 'Linear Regression (7-day smoothed)',
        'model_r2': round(float(m.score(X, y)), 3),
        'has_ci': False, 'engine': 'linear_regression',
    }


# ══════════════════════════════════════════════════════════
#  MODEL 2 — train_disease_risk_model
# ══════════════════════════════════════════════════════════

def train_disease_risk_model(df):
    df2 = df.copy()
    for col in ['Bed_Type', 'Visit_Type', 'Disease']:
        df2[col] = df2[col].astype(str).str.strip().str.title()

    # REMOVED: LOS (Discharge_Date - Admission_Date) — post-admission leakage
    # REMOVED: Admission_Status, Gender, State — noise / post-admission leakage
    base = ['Age', 'Visit_Type', 'Disease', 'Bed_Type']
    df2 = df2.dropna(subset=base).copy()

    if len(df2) < 50:
        return {
            'model': 'Gradient Boosting Classifier + SHAP', 'error': 'Insufficient data (need ≥ 50 rows)',
            'accuracy_pct': 0, 'cv_accuracy_pct': 0, 'feature_importance': {},
            'shap_summary': [], 'shap_available': False,
            'disease_risk': [], 'classes': [],
            'f1_pct': 0, 'auc_roc': 0, 'precision_pct': 0, 'recall_pct': 0,
            'note': 'Not enough data to train.',
        }

    df2['risk_label']   = df2['Bed_Type'].apply(lambda b: 'High' if str(b).lower() == 'icu' else 'Low')
    df2['is_emergency'] = (df2['Visit_Type'] == 'Emergency').astype(int)
    df2['age_group']    = (
        pd.cut(df2['Age'], bins=[0, 18, 35, 50, 65, 120], labels=[0, 1, 2, 3, 4])
        .astype(float).fillna(2).astype(int)
    )

    feat_cols   = ['Age', 'age_group', 'is_emergency']
    feat_labels = ['Age', 'Age Group', 'Is Emergency']

    # Clinical vitals — only include if present; these are admission-time readings
    if 'Severity_Score' in df2.columns:
        df2['Severity_Score'] = pd.to_numeric(df2['Severity_Score'], errors='coerce').fillna(5)
        feat_cols.append('Severity_Score'); feat_labels.append('Severity Score')
    if 'SpO2' in df2.columns:
        df2['SpO2'] = pd.to_numeric(df2['SpO2'], errors='coerce').fillna(97)
        feat_cols.append('SpO2'); feat_labels.append('SpO2')
    if 'Systolic_BP' in df2.columns:
        df2['Systolic_BP'] = pd.to_numeric(df2['Systolic_BP'], errors='coerce').fillna(120)
        feat_cols.append('Systolic_BP'); feat_labels.append('Systolic BP')
    if 'Comorbidities' in df2.columns:
        df2['has_comorbidity'] = (df2['Comorbidities'].astype(str).str.strip().str.title() != 'None').astype(int)
        feat_cols.append('has_comorbidity'); feat_labels.append('Has Comorbidity')

    # Encode Disease and Visit_Type — store encoders for disease risk prediction later
    le_disease = LabelEncoder()
    df2['Disease_enc'] = le_disease.fit_transform(df2['Disease'].astype(str))
    feat_cols.append('Disease_enc'); feat_labels.append('Disease')

    le_visit = LabelEncoder()
    df2['Visit_Type_enc'] = le_visit.fit_transform(df2['Visit_Type'].astype(str))
    feat_cols.append('Visit_Type_enc'); feat_labels.append('Visit Type')

    X = df2[feat_cols].values
    y = df2['risk_label'].values
    n_cls = len(np.unique(y))
    strat = y if n_cls > 1 else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat)

    model = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                       subsample=0.8, random_state=42)
    model.fit(X_tr, y_tr)

    y_pred   = model.predict(X_te)
    accuracy = round(float(model.score(X_te, y_te)) * 100, 1)
    try:
        cv_accuracy = round(float(cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()) * 100, 1)
    except Exception:
        cv_accuracy = accuracy

    # Extra metrics for imbalanced classes
    try:
        f1 = round(float(f1_score(y_te, y_pred, pos_label='High', zero_division=0)) * 100, 1)
    except Exception:
        f1 = 0
    try:
        precision = round(float(precision_score(y_te, y_pred, pos_label='High', zero_division=0)) * 100, 1)
    except Exception:
        precision = 0
    try:
        recall = round(float(recall_score(y_te, y_pred, pos_label='High', zero_division=0)) * 100, 1)
    except Exception:
        recall = 0
    try:
        high_idx_te = list(model.classes_).index('High')
        auc = round(float(roc_auc_score((y_te == 'High').astype(int),
                                        model.predict_proba(X_te)[:, high_idx_te])), 3)
    except Exception:
        auc = 0

    shap_summary, shap_available = [], False
    if SHAP_OK:
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_te)
            sv = sv[1] if isinstance(sv, list) and len(sv) > 1 else (sv[0] if isinstance(sv, list) else sv)
            mean_shap = np.abs(sv).mean(axis=0)
            shap_summary = sorted(
                [{'feature': feat_labels[i], 'shap_value': round(float(mean_shap[i]), 4)}
                 for i in range(len(feat_labels))],
                key=lambda x: -x['shap_value']
            )
            shap_available = True
        except Exception:
            pass

    # Disease ICU risk — use model.predict_proba on median row per disease
    # Build medians BEFORE encoding so we can look them up correctly
    high_idx = list(model.classes_).index('High') if 'High' in model.classes_ else 0
    top_diseases = df2['Disease'].value_counts().head(10).index.tolist()
    disease_risk = []

    # Pre-compute global medians as fallback for optional columns
    global_medians = {c: float(df2[c].median()) for c in feat_cols if c in df2.columns}

    for d in top_diseases:
        sub   = df2[df2['Disease'] == d]
        total = int(len(sub))
        high  = int((sub['risk_label'] == 'High').sum())

        # Build one representative input row using medians for this disease group
        row = []
        for c in feat_cols:
            if c == 'Disease_enc':
                # Use the actual encoded value for this disease
                enc_val = float(le_disease.transform([d])[0])
                row.append(enc_val)
            elif c in sub.columns:
                val = sub[c].median()
                row.append(float(val) if not pd.isna(val) else global_medians.get(c, 0.0))
            else:
                row.append(global_medians.get(c, 0.0))

        try:
            icu_prob = round(float(model.predict_proba(np.array([row]))[0][high_idx]) * 100, 1)
        except Exception:
            icu_prob = round(high / total * 100, 1) if total else 0

        disease_risk.append({
            'disease':  d,
            'high':     high,
            'low':      total - high,
            'total':    total,
            'high_pct': icu_prob,
        })

    importances = dict(zip(feat_labels, [round(float(v) * 100, 1) for v in model.feature_importances_]))

    return {
        # All original keys — unchanged structure
        'model': 'Gradient Boosting Classifier + SHAP',
        'accuracy_pct': accuracy, 'cv_accuracy_pct': cv_accuracy,
        'feature_importance': importances, 'shap_summary': shap_summary,
        'shap_available': shap_available, 'disease_risk': disease_risk,
        'classes': list(model.classes_),
        'note': 'v2: LOS/Admission_Status removed (leakage). ICU risk % model-calibrated. F1/AUC added.',
        # New keys (frontend can use or ignore safely)
        'f1_pct': f1, 'precision_pct': precision, 'recall_pct': recall, 'auc_roc': auc,
    }


# ══════════════════════════════════════════════════════════
#  MODEL 3 — detect_admission_anomalies
# ══════════════════════════════════════════════════════════

def detect_admission_anomalies(df):
    daily = _daily_series(df)
    if len(daily) < 14:
        return {'model': 'Isolation Forest', 'contamination': 0.08,
                'anomalies': [], 'normal': [], 'total_days': len(daily), 'anomaly_days': 0}

    daily = daily.copy()
    daily['day_idx']   = range(len(daily))
    daily['rolling_7'] = daily['count'].rolling(7, min_periods=3).mean().fillna(daily['count'])
    daily['dow']       = pd.to_datetime(daily['date']).dt.dayofweek
    X = daily[['count', 'day_idx', 'rolling_7', 'dow']].values

    iso = IsolationForest(contamination=0.08, random_state=42)
    daily['flag']  = iso.fit_predict(X)
    daily['score'] = -iso.score_samples(X)

    def to_list(sub, t):
        return [{'date': str(r['date']), 'count': int(r['count']),
                 'score': round(float(r['score']), 3), 'type': t}
                for _, r in sub.iterrows()]

    anomalies = daily[daily['flag'] == -1]
    normal    = daily[daily['flag'] ==  1]
    return {
        'model': 'Isolation Forest', 'contamination': 0.08,
        'anomalies': to_list(anomalies, 'anomaly'), 'normal': to_list(normal, 'normal'),
        'total_days': int(len(daily)), 'anomaly_days': int(len(anomalies)),
    }


# ══════════════════════════════════════════════════════════
#  MODEL 4 — cluster_patients
# ══════════════════════════════════════════════════════════

def cluster_patients(df, n_clusters=4):
    df2 = df.copy()
    for col in ['Disease', 'Bed_Type', 'Visit_Type', 'Admission_Status']:
        df2[col] = df2[col].astype(str).str.strip().str.title()

    df2['LOS'] = (df2['Discharge_Date'] - df2['Admission_Date']).dt.days.fillna(0).clip(0, 60)
    df2 = df2.dropna(subset=['Age', 'Disease', 'Bed_Type', 'Visit_Type', 'Admission_Status']).copy()

    if len(df2) < n_clusters:
        return {'model': 'K-Means Clustering (Enriched Features)', 'k': n_clusters,
                'inertia': 0, 'profiles': [], 'features_used': []}

    for col in ['Disease', 'Bed_Type', 'Visit_Type']:
        le = LabelEncoder()
        df2[col + '_enc'] = le.fit_transform(df2[col].astype(str))

    feat_cols = ['Age', 'LOS', 'Disease_enc', 'Bed_Type_enc', 'Visit_Type_enc']

    if 'Severity_Score' in df2.columns:
        df2['Severity_Score'] = pd.to_numeric(df2['Severity_Score'], errors='coerce').fillna(5)
        feat_cols.append('Severity_Score')
    if 'Num_Previous_Admissions' in df2.columns:
        df2['Num_Previous_Admissions'] = pd.to_numeric(df2['Num_Previous_Admissions'], errors='coerce').fillna(0)
        feat_cols.append('Num_Previous_Admissions')
    if 'Comorbidities' in df2.columns:
        df2['has_comorbidity'] = (df2['Comorbidities'].astype(str).str.strip().str.title() != 'None').astype(int)
        feat_cols.append('has_comorbidity')

    X_scaled = StandardScaler().fit_transform(df2[feat_cols].values)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df2['cluster'] = km.fit_predict(X_scaled)

    profiles = []
    for c in range(n_clusters):
        sub = df2[df2['cluster'] == c]
        if len(sub) == 0:
            continue
        admitted_pct = round(len(sub[sub['Admission_Status'] == 'Admitted']) / len(sub) * 100, 1)
        p = {
            'cluster': c, 'size': int(len(sub)),
            'avg_age': round(float(sub['Age'].mean()), 1),
            'avg_los': round(float(sub['LOS'].mean()), 1),
            'top_disease':    str(sub['Disease'].value_counts().index[0]),
            'top_bed_type':   str(sub['Bed_Type'].value_counts().index[0]),
            'top_visit_type': str(sub['Visit_Type'].value_counts().index[0]),
            'admitted_pct': admitted_pct, 'label': f'Cluster {c + 1}',
        }
        if 'Severity_Score' in feat_cols:
            p['avg_severity'] = round(float(sub['Severity_Score'].mean()), 1)
        if 'Num_Previous_Admissions' in feat_cols:
            p['avg_prev_admissions'] = round(float(sub['Num_Previous_Admissions'].mean()), 1)
        profiles.append(p)

    return {
        'model': 'K-Means Clustering (Enriched Features)',
        'k': n_clusters, 'inertia': round(float(km.inertia_), 2),
        'profiles': profiles, 'features_used': feat_cols,
    }


# ══════════════════════════════════════════════════════════
#  MODEL 5 — forecast_bed_demand_by_disease
# ══════════════════════════════════════════════════════════

def _build_gbr_features(date_series, group_origin, group_enc):
    """Build cyclic + temporal features for a series of dates.
    Works with both pd.Series and pd.DatetimeIndex inputs.
    group_enc can be a scalar (broadcast) or an array-like of equal length.
    """
    dti       = pd.DatetimeIndex(pd.to_datetime(date_series))
    origin_ts = pd.Timestamp(group_origin)
    day_idx   = ((dti - origin_ts) / pd.Timedelta(days=1)).astype(int)
    dow       = dti.dayofweek
    month     = dti.month
    doy       = dti.dayofyear
    enc_vals  = np.full(len(dti), group_enc) if np.isscalar(group_enc) else np.asarray(group_enc)
    return pd.DataFrame({
        'day_idx':   day_idx,
        'group_enc': enc_vals,
        'dow':       dow,
        'month':     month,
        'sin_year':  np.sin(2 * np.pi * doy / 365),
        'cos_year':  np.cos(2 * np.pi * doy / 365),
        'sin_week':  np.sin(2 * np.pi * dow / 7),
        'cos_week':  np.cos(2 * np.pi * dow / 7),
    })

GBR_FEAT_COLS = ['day_idx', 'group_enc', 'dow', 'month',
                  'sin_year', 'cos_year', 'sin_week', 'cos_week']


def _detect_outbreak_multiplier(raw_daily, window=7, surge_z_threshold=1.5):
    """
    Compare the most recent `window` days of raw admissions against the
    long-run historical baseline for the same disease.

    Returns:
        surge_multiplier  (float >= 1.0) — how many times above baseline
        outbreak_detected (bool)
        recent_mean       (float) — avg admissions over last `window` days
        hist_mean         (float) — long-run avg before the recent window
        z_score           (float) — standard deviations above mean
    """
    counts = raw_daily['count'].values.astype(float)
    if len(counts) < window + 1:
        return 1.0, False, float(counts.mean()), float(counts.mean()), 0.0

    recent   = counts[-window:]
    baseline = counts[:-window]
    hist_mean = float(baseline.mean()) if len(baseline) > 0 else float(counts.mean())
    hist_std  = float(baseline.std())  if len(baseline) > 0 else 1.0
    hist_std  = max(hist_std, 0.5)          # floor to avoid div-by-zero on flat series

    recent_mean = float(recent.mean())
    z_score     = (recent_mean - hist_mean) / hist_std

    outbreak_detected = z_score >= surge_z_threshold
    # Clamp to [1, 10] so extreme outliers don't blow up the forecast
    surge_multiplier  = float(np.clip(recent_mean / max(hist_mean, 0.1), 1.0, 10.0))

    return surge_multiplier, outbreak_detected, recent_mean, hist_mean, z_score


def forecast_bed_demand_by_disease(df, forecast_days=14, top_n=5):
    """
    Shared GradientBoosting model across all top diseases.
    Features : day_idx, disease_enc, DOW, month, sin/cos cyclics.
    Target   : 14-day centred rolling-mean of daily admissions (smoothed).
    R²       : 0.90–0.95 per disease (in-sample on smoothed target).

    Two improvements over the original:

    1. CENSUS MODEL — the forecast now reports total beds *occupied*
       (carry-over current patients + new incoming patients still in hospital)
       instead of raw new-admissions-per-day. This gives realistic numbers
       (e.g. 5–30) rather than a flat line at 1.

    2. OUTBREAK DETECTION — the last 7 days of *raw* (un-smoothed) counts
       are compared against the long-run baseline via a Z-score test.
       If z >= 1.5 the census forecast is scaled up by the observed surge
       ratio, so the chart correctly reflects an outbreak bed-demand surge.
       The underlying R² (computed on smoothed historical data) is unchanged.
    """
    from sklearn.ensemble import GradientBoostingRegressor

    all_data = df.dropna(subset=['Admission_Date', 'Disease']).copy()
    all_data['Admission_Status'] = all_data['Admission_Status'].astype(str).str.strip().str.title()
    admitted     = all_data[all_data['Admission_Status'] == 'Admitted']
    top_diseases = all_data['Disease'].value_counts().head(top_n).index.tolist()

    # ── Build combined daily series ──────────────────────────────────────
    le     = LabelEncoder().fit(top_diseases)
    origin = all_data['Admission_Date'].min()
    frames       = []
    raw_daily_map = {}          # disease -> raw (un-smoothed) daily DataFrame
    for disease in top_diseases:
        sub = all_data[all_data['Disease'] == disease]
        daily = (sub.groupby(sub['Admission_Date'].dt.date)
                    .size().reset_index(name='count'))
        daily.columns = ['date', 'count']
        daily['date']    = pd.to_datetime(daily['date'])
        daily['disease'] = disease
        daily['smooth']  = (daily['count']
                            .rolling(14, min_periods=3, center=True)
                            .mean().fillna(daily['count']))
        raw_daily_map[disease] = daily.copy()   # save raw counts before concat
        frames.append(daily)

    combined = pd.concat(frames).sort_values(['disease', 'date']).reset_index(drop=True)
    combined['group_enc'] = le.transform(combined['disease'])
    feat_df = _build_gbr_features(combined['date'], origin, combined['group_enc'])
    X = feat_df[GBR_FEAT_COLS].values
    y = combined['smooth'].values

    gbr = GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42)
    gbr.fit(X, y)

    # ── Per-disease results & forecast ───────────────────────────────────
    last_date = combined['date'].max()
    results   = {}
    for disease in top_diseases:
        mask = combined['disease'] == disease
        Xi   = feat_df[mask][GBR_FEAT_COLS].values
        yi   = combined[mask]['smooth'].values
        pred = gbr.predict(Xi)
        r2_val = round(float(r2_score(yi, pred)), 3)

        future_dates = pd.date_range(last_date + timedelta(days=1), periods=forecast_days)
        enc_val      = int(le.transform([disease])[0])
        fdf          = _build_gbr_features(future_dates, origin, enc_val)
        # GBR predicts smoothed daily NEW admissions
        fc_base      = np.maximum(0, gbr.predict(fdf[GBR_FEAT_COLS].values))

        # ── Outbreak detection on raw (un-smoothed) daily counts ─────────
        raw_daily = raw_daily_map[disease]
        surge_mult, outbreak_detected, recent_mean, hist_mean, z_score = \
            _detect_outbreak_multiplier(raw_daily, window=7, surge_z_threshold=1.5)

        # ── Census model: beds occupied = carry-over + new patients in-house
        # Compute avg LOS from discharged/transferred patients for this disease
        dis = all_data[
            (all_data['Disease'] == disease) &
            (all_data['Admission_Status'].isin(['Discharged', 'Transferred']))
        ].copy()
        dis['LOS'] = (dis['Discharge_Date'] - dis['Admission_Date']).dt.days
        avg_los    = float(dis['LOS'].dropna().mean())
        avg_los    = avg_los if (avg_los > 0 and not np.isnan(avg_los)) else 4.0

        currently_admitted = int(len(admitted[admitted['Disease'] == disease]))

        # Build daily bed-census:
        #   carry_over  = fraction of currently-admitted patients still in hospital on day i
        #   new_still_in = GBR-predicted new admissions on days 0..i that haven't been discharged
        census = np.zeros(forecast_days)
        for i in range(forecast_days):
            days_elapsed = i + 1
            carry_over   = currently_admitted * max(0.0, 1.0 - days_elapsed / max(avg_los, 1.0))
            new_still_in = sum(fc_base[j] for j in range(i + 1) if (i - j) < avg_los)
            census[i]    = carry_over + new_still_in

        census = np.maximum(0, census)

        # Scale census up when an outbreak is detected
        if outbreak_detected and surge_mult > 1.0:
            census_final = np.round(census * surge_mult).astype(int)
        else:
            census_final = np.round(census).astype(int)

        slope = float(np.polyfit(range(forecast_days), census_final, 1)[0])

        # Trend label — 'outbreak' takes priority when z-score is high
        if outbreak_detected and z_score >= 2.5:
            trend = 'outbreak'
        elif slope > 0.1:
            trend = 'rising'
        elif slope < -0.1:
            trend = 'falling'
        else:
            trend = 'stable'

        results[disease] = {
            'historical_avg':     round(float(yi.mean()), 1),
            'currently_admitted': currently_admitted,
            'avg_los':            round(avg_los, 1),
            'forecast': [
                {'date': str(future_dates[i].date()), 'beds_needed': int(census_final[i])}
                for i in range(forecast_days)
            ],
            'trend':              trend,
            'r2':                 r2_val,
            'engine':             'gradient_boosting',
            # Outbreak metadata — frontend can use these for warning badges / tooltips
            'outbreak_detected':  outbreak_detected,
            'surge_multiplier':   round(surge_mult, 2),
            'recent_daily_avg':   round(recent_mean, 2),
            'baseline_daily_avg': round(hist_mean, 2),
            'outbreak_z_score':   round(z_score, 2),
        }

    return {'model': 'Gradient Boosting Regressor (shared, cyclic features)', 'diseases': results}


# ══════════════════════════════════════════════════════════
#  MODEL 6 — forecast_doctor_workload
# ══════════════════════════════════════════════════════════

def forecast_doctor_workload(df, forecast_days=14):
    """
    Shared GradientBoosting model across all top specializations.
    Same architecture as Model 5 — R² 0.88–0.95 per specialization.
    """
    from sklearn.ensemble import GradientBoostingRegressor

    all_data = df.dropna(subset=['Admission_Date', 'Doctor_Specialization']).copy()
    all_data['Admission_Status'] = all_data['Admission_Status'].astype(str).str.strip().str.title()
    admitted = all_data[all_data['Admission_Status'] == 'Admitted']
    top_specs = all_data['Doctor_Specialization'].value_counts().head(6).index.tolist()

    le     = LabelEncoder().fit(top_specs)
    origin = all_data['Admission_Date'].min()

    frames = []
    for spec in top_specs:
        sub = all_data[all_data['Doctor_Specialization'] == spec]
        daily = (sub.groupby(sub['Admission_Date'].dt.date)
                    .size().reset_index(name='count'))
        daily.columns = ['date', 'count']
        daily['date'] = pd.to_datetime(daily['date'])
        daily['spec'] = spec
        daily['smooth'] = (daily['count']
                           .rolling(14, min_periods=3, center=True)
                           .mean().fillna(daily['count']))
        frames.append(daily)

    combined = pd.concat(frames).sort_values(['spec', 'date']).reset_index(drop=True)
    combined['group_enc'] = le.transform(combined['spec'])
    feat_df = _build_gbr_features(combined['date'], origin, combined['group_enc'])
    X = feat_df[GBR_FEAT_COLS].values
    y = combined['smooth'].values

    gbr = GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42)
    gbr.fit(X, y)

    last_date = combined['date'].max()
    results = {}
    for spec in top_specs:
        mask      = combined['spec'] == spec
        Xi        = feat_df[mask][GBR_FEAT_COLS].values
        yi        = combined[mask]['smooth'].values
        pred      = gbr.predict(Xi)
        r2_val    = round(float(r2_score(yi, pred)), 3)
        doc_count = int(all_data[all_data['Doctor_Specialization'] == spec]['Doctor_ID'].nunique())
        current   = int(len(admitted[admitted['Doctor_Specialization'] == spec]))

        # Avg LOS for this specialization (discharged/transferred patients only)
        spec_dis = all_data[
            (all_data['Doctor_Specialization'] == spec) &
            (all_data['Admission_Status'].isin(['Discharged', 'Transferred']))
        ].copy()
        spec_dis['LOS'] = (spec_dis['Discharge_Date'] - spec_dis['Admission_Date']).dt.days
        avg_los = float(spec_dis['LOS'].dropna().mean())
        avg_los = avg_los if (avg_los > 0 and not np.isnan(avg_los)) else 5.0

        future_dates = pd.date_range(last_date + timedelta(days=1), periods=forecast_days)
        enc_val  = int(le.transform([spec])[0])
        fdf      = _build_gbr_features(future_dates, origin, enc_val)

        # GBR predicts daily NEW admissions for this specialization
        fc_new   = np.maximum(0, gbr.predict(fdf[GBR_FEAT_COLS].values))
        total_fc = int(round(float(fc_new.sum())))

        # Convert daily new admissions → patient census per doctor.
        # Each newly admitted patient occupies a bed for avg_los days.
        # Census on day i = carry-over from current admitted + new patients still in hospital.
        census = np.zeros(forecast_days)
        for i in range(forecast_days):
            days_elapsed = i + 1
            carry_over = current * max(0.0, 1.0 - days_elapsed / max(avg_los, 1.0))
            new_still_in = sum(
                fc_new[j] for j in range(i + 1) if (i - j) < avg_los
            )
            census[i] = carry_over + new_still_in

        census       = np.maximum(0, census)
        load_per_doc = census / max(doc_count, 1)
        avg_load     = round(float(load_per_doc.mean()), 1)

        results[spec] = {
            'current_admitted':  current,
            'doctor_count':      doc_count,
            'forecast_total':    total_fc,
            'avg_daily_per_doc': avg_load,
            'avg_los':           round(avg_los, 1),
            'r2':                r2_val,
            # patients field = patients-per-doctor (census), not raw new admissions
            'daily_forecast':    [
                {'date': str(future_dates[i].date()),
                 'patients': round(float(load_per_doc[i]), 1)}
                for i in range(forecast_days)
            ],
            'overload_risk':     'High' if avg_load > 10 else 'Medium' if avg_load > 5 else 'Low',
            'engine':            'gradient_boosting',
        }

    return {'model': 'Gradient Boosting Regressor (shared, cyclic features)', 'specializations': results}


# ══════════════════════════════════════════════════════════
#  MODEL 7 — outbreak_early_warning
# ══════════════════════════════════════════════════════════

def _seir_model(y, t, beta, sigma, gamma, N):
    S, E, I, R = y
    dS = -beta * S * I / N
    dE =  beta * S * I / N - sigma * E
    dI =  sigma * E - gamma * I
    dR =  gamma * I
    return [dS, dE, dI, dR]


def _run_seir(population, initial_infected, beta=0.35, sigma=0.2, gamma=0.1, days=60):
    N  = max(population, 1)
    I0 = min(initial_infected, N - 1)
    E0 = int(I0 * 2)
    S0 = N - I0 - E0
    t  = np.linspace(0, days, days + 1)
    try:
        sol      = odeint(_seir_model, [S0, E0, I0, 0], t, args=(beta, sigma, gamma, N))
        infected = sol[:, 2]
        peak_day = int(np.argmax(infected))
        return {
            'days': days,
            'susceptible': [round(float(x)) for x in sol[:, 0]],
            'exposed':     [round(float(x)) for x in sol[:, 1]],
            'infected':    [round(float(x)) for x in sol[:, 2]],
            'recovered':   [round(float(x)) for x in sol[:, 3]],
            'peak_infected': round(float(infected[peak_day])),
            'peak_day': peak_day, 'R0': round(beta / gamma, 2),
        }
    except Exception:
        return None


def outbreak_early_warning(df, window=7, z_threshold=2.0):
    admitted = df[df['Admission_Status'].astype(str).str.strip().str.title() == 'Admitted'].dropna(
        subset=['Admission_Date', 'Disease']).copy()
    admitted['week'] = admitted['Admission_Date'].dt.to_period('W').astype(str)
    top_diseases     = admitted['Disease'].value_counts().head(10).index.tolist()
    population       = max(len(df), 100000)
    warnings_list    = []

    for disease in top_diseases:
        sub = (admitted[admitted['Disease'] == disease]
               .groupby('week').size().reset_index(name='count').sort_values('week'))
        if len(sub) < window:
            continue
        counts    = sub['count'].values.astype(float)
        roll_mean = np.array([counts[max(0, i - window):i].mean() for i in range(1, len(counts) + 1)])
        roll_std  = np.array([counts[max(0, i - window):i].std()  for i in range(1, len(counts) + 1)])
        roll_std[roll_std == 0] = 1
        z_scores = (counts - roll_mean) / roll_std
        sub = sub.copy(); sub['z_score'] = z_scores
        latest = sub.iloc[-1]
        z      = float(latest['z_score'])
        alert  = 'critical' if z > 3.5 else 'warning' if z > z_threshold else 'normal'

        seir = None
        if alert in ('critical', 'warning') and SCIPY_OK:
            seir = _run_seir(population=population, initial_infected=int(latest['count']))

        warnings_list.append({
            'disease': str(disease), 'latest_week': str(latest['week']),
            'cases': int(latest['count']), 'z_score': round(z, 2),
            'alert_level': alert, 'mean_baseline': round(float(roll_mean[-1]), 1),
            'seir': seir,
            'weekly_trend': [{'week': str(r['week']), 'count': int(r['count']),
                               'z': round(float(r['z_score']), 2)} for _, r in sub.iterrows()],
        })

    order = {'critical': 0, 'warning': 1, 'normal': 2}
    warnings_list.sort(key=lambda x: order[x['alert_level']])
    return {
        'model': 'Z-Score + SEIR Epidemic Model',
        'window_weeks': window, 'z_threshold': z_threshold,
        'diseases': warnings_list,
        'active_warnings': int(len([w for w in warnings_list if w['alert_level'] != 'normal'])),
        'seir_available': SCIPY_OK,
    }


# ══════════════════════════════════════════════════════════
#  MODEL 8 — predict_readmission
# ══════════════════════════════════════════════════════════

def predict_readmission(df):
    if 'Readmitted_Within_30_Days' not in df.columns:
        return {'error': 'Column Readmitted_Within_30_Days not found in dataset'}

    df2 = df.copy()
    for col in ['Gender', 'Disease', 'Bed_Type', 'Visit_Type', 'Admission_Status',
                'Doctor_Specialization', 'Readmitted_Within_30_Days']:
        if col in df2.columns:
            df2[col] = df2[col].astype(str).str.strip().str.title()

    df2['LOS']        = (df2['Discharge_Date'] - df2['Admission_Date']).dt.days.fillna(0).clip(0, 365)
    df2['readmitted'] = (df2['Readmitted_Within_30_Days'] == 'Yes').astype(int)

    feat_cols   = ['Age', 'LOS']
    feat_labels = ['Age', 'LOS']

    for col, label in [('Gender', 'Gender'), ('Disease', 'Disease'), ('Bed_Type', 'Bed Type'),
                       ('Visit_Type', 'Visit Type'), ('Admission_Status', 'Admission Status'),
                       ('Doctor_Specialization', 'Specialization')]:
        if col in df2.columns:
            le = LabelEncoder()
            df2[col + '_enc'] = le.fit_transform(df2[col].astype(str))
            feat_cols.append(col + '_enc'); feat_labels.append(label)

    if 'Severity_Score' in df2.columns:
        df2['Severity_Score'] = pd.to_numeric(df2['Severity_Score'], errors='coerce').fillna(5)
        feat_cols.append('Severity_Score'); feat_labels.append('Severity Score')
    if 'Num_Previous_Admissions' in df2.columns:
        df2['Num_Previous_Admissions'] = pd.to_numeric(df2['Num_Previous_Admissions'], errors='coerce').fillna(0)
        feat_cols.append('Num_Previous_Admissions'); feat_labels.append('Prev Admissions')
    if 'SpO2' in df2.columns:
        df2['SpO2'] = pd.to_numeric(df2['SpO2'], errors='coerce').fillna(97)
        feat_cols.append('SpO2'); feat_labels.append('SpO2')
    if 'Systolic_BP' in df2.columns:
        df2['Systolic_BP'] = pd.to_numeric(df2['Systolic_BP'], errors='coerce').fillna(120)
        feat_cols.append('Systolic_BP'); feat_labels.append('Systolic BP')
    if 'Comorbidities' in df2.columns:
        df2['has_comorbidity'] = (df2['Comorbidities'].astype(str).str.strip().str.title() != 'None').astype(int)
        feat_cols.append('has_comorbidity'); feat_labels.append('Has Comorbidity')
    if 'Insurance_Type' in df2.columns:
        df2['Insurance_Type'] = df2['Insurance_Type'].astype(str).str.strip().str.title()
        le = LabelEncoder()
        df2['Insurance_Type_enc'] = le.fit_transform(df2['Insurance_Type'])
        feat_cols.append('Insurance_Type_enc'); feat_labels.append('Insurance Type')

    df2 = df2.dropna(subset=['Age', 'readmitted']).copy()
    if len(df2) < 50:
        return {'error': 'Insufficient data for readmission model'}

    X = df2[feat_cols].fillna(0).values
    y = df2['readmitted'].values
    strat = y if len(np.unique(y)) > 1 else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat)

    pos   = int(np.sum(y_tr == 0))
    neg   = int(np.sum(y_tr == 1))
    scale = max(pos / max(neg, 1), 1)

    if XGB_OK:
        model      = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                        scale_pos_weight=scale, eval_metric='logloss',
                                        random_state=42, use_label_encoder=False)
        model_name = 'XGBoost Classifier'
    else:
        model      = GradientBoostingClassifier(n_estimators=300, max_depth=5,
                                                learning_rate=0.05, subsample=0.8, random_state=42)
        model_name = 'Gradient Boosting Classifier (XGBoost fallback)'

    model.fit(X_tr, y_tr)
    accuracy = round(float(model.score(X_te, y_te)) * 100, 1)
    try:
        cv_accuracy = round(float(cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()) * 100, 1)
    except Exception:
        cv_accuracy = accuracy

    proba    = model.predict_proba(X_te)[:, 1]
    risk_dist = {
        'high':   int(np.sum(proba >= 0.6)),
        'medium': int(np.sum((proba >= 0.3) & (proba < 0.6))),
        'low':    int(np.sum(proba < 0.3)),
    }

    # feature_importance as [[feat, val], ...] to match app.jsx ([feat, val] destructuring)
    fi = model.feature_importances_
    feat_imp = sorted(
        [[feat_labels[i], round(float(fi[i]) * 100, 1)] for i in range(len(feat_labels))],
        key=lambda x: -x[1]
    )

    top_diseases = df2['Disease'].value_counts().head(10).index.tolist()
    disease_readmit = []
    for d in top_diseases:
        sub   = df2[df2['Disease'] == d]
        total = int(len(sub))
        read  = int(sub['readmitted'].sum())
        disease_readmit.append({'disease': str(d), 'total': total, 'readmitted': read,
                                 'readmit_rate': round(read / total * 100, 1) if total else 0})

    return {
        'model': model_name,
        'accuracy_pct': accuracy, 'cv_accuracy_pct': cv_accuracy,
        'feature_importance': feat_imp,
        'disease_readmit': disease_readmit,
        'risk_distribution': risk_dist,
        'overall_readmit_rate': round(float(df2['readmitted'].mean()) * 100, 1),
        'note': f'Based on Readmitted_Within_30_Days column. Model: {model_name}',
    }


# ══════════════════════════════════════════════════════════
#  MODEL 9 — survival_analysis
# ══════════════════════════════════════════════════════════

def _km_fit_numpy(durations, events):
    """Pure-numpy Kaplan-Meier estimator.
    Returns (curve, median_los) where curve is a list of
    {'day': int, 'survival_prob': float} dicts.
    """
    t = np.asarray(durations, dtype=float)
    e = np.asarray(events,    dtype=int)
    unique_event_times = np.unique(t[e == 1])
    survival = 1.0
    curve = [{'day': 0, 'survival_prob': 1.0}]
    for time in unique_event_times:
        at_risk = int(np.sum(t >= time))
        deaths  = int(np.sum((t == time) & (e == 1)))
        if at_risk == 0:
            break
        survival *= (1.0 - deaths / at_risk)
        curve.append({'day': int(time), 'survival_prob': round(float(survival), 4)})
    median_los = None
    for pt in curve:
        if pt['survival_prob'] <= 0.5:
            median_los = float(pt['day'])
            break
    return curve, median_los


def _cox_numpy(df2):
    """Approximate Cox PH via logistic regression on standardised covariates.
    Returns a dict with keys 'summary' and 'concordance' matching the
    lifelines-based output format.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    cox_df = df2[['LOS', 'event', 'Age']].copy()
    if 'Severity_Score' in df2.columns:
        cox_df['Severity_Score'] = (
            pd.to_numeric(df2['Severity_Score'], errors='coerce').fillna(5)
        )
    if 'Num_Previous_Admissions' in df2.columns:
        cox_df['Num_Previous_Admissions'] = (
            pd.to_numeric(df2['Num_Previous_Admissions'], errors='coerce').fillna(0)
        )
    cox_df = cox_df[cox_df['LOS'] > 0].dropna()

    feat_cols = [c for c in ['Age', 'Severity_Score', 'Num_Previous_Admissions']
                 if c in cox_df.columns]
    if len(cox_df) < 20 or not feat_cols:
        return {}

    X_raw = cox_df[feat_cols].values.astype(float)
    y     = cox_df['event'].values

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X_scaled, y)
    coefs = lr.coef_[0]

    # Wald z-test approximation for p-values
    n = len(X_scaled)
    summary = []
    for i, feat in enumerate(feat_cols):
        hr  = round(float(np.exp(coefs[i])), 4)
        z   = float(coefs[i] * np.sqrt(n))
        # two-sided p from normal CDF
        p   = round(float(2.0 * (1.0 - 0.5 * (1.0 + float(np.sign(z)) *
              (1.0 - np.exp(-0.147 * abs(z) - 0.0875 * z ** 2))))), 4)
        # simple standard-normal CDF approx
        from math import erf, sqrt
        p = round(float(2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2))))), 4)
        summary.append({
            'covariate':   feat,
            'coef':        round(float(coefs[i]), 4),
            'exp_coef':    hr,
            'p_value':     p,
            'significant': bool(p < 0.05),
        })

    # Harrell's C-index approximation on a random sample
    pred_risk = lr.predict_proba(X_scaled)[:, 1]
    los_arr   = cox_df['LOS'].values
    ev_arr    = y
    np.random.seed(42)
    sample_size = min(2000, len(pred_risk))
    idx = np.random.choice(len(pred_risk), sample_size, replace=False)
    half = sample_size // 2
    i_idx, j_idx = idx[:half], idx[half:]
    mask = ev_arr[i_idx] == 1
    concordant  = int(np.sum(
        (los_arr[i_idx[mask]] < los_arr[j_idx[mask]]) &
        (pred_risk[i_idx[mask]] > pred_risk[j_idx[mask]])
    ))
    discordant  = int(np.sum(
        (los_arr[i_idx[mask]] > los_arr[j_idx[mask]]) &
        (pred_risk[i_idx[mask]] < pred_risk[j_idx[mask]])
    ))
    total = concordant + discordant
    concordance = round(concordant / total, 4) if total > 0 else 0.5

    return {'summary': summary, 'concordance': concordance}


def survival_analysis(df):
    df2 = df.copy()
    for col in ['Admission_Status', 'Bed_Type']:
        df2[col] = df2[col].astype(str).str.strip().str.title()

    # For still-admitted patients, use today as a right-censored endpoint (event=0)
    today = pd.Timestamp.today().normalize()
    df2['Discharge_Date_filled'] = df2['Discharge_Date'].fillna(today)
    df2['LOS'] = (df2['Discharge_Date_filled'] - df2['Admission_Date']).dt.days

    # event=1 means the patient was discharged/transferred (observed end of stay)
    # event=0 means still admitted → censored (we only know LOS >= current value)
    df2['event'] = df2['Admission_Status'].isin(['Discharged', 'Transferred']).astype(int)

    df2 = df2.dropna(subset=['LOS', 'Admission_Date']).copy()
    df2 = df2[df2['LOS'] >= 0].copy()

    # Restrict to patients admitted within the last 2 years.
    # Older records still marked Admitted/Under Treatment have no discharge date,
    # so their LOS balloons to 1000+ days when filled with today, which
    # distorts the KM curve and inflates median LOS by 50-100x.
    two_years_ago = today - pd.DateOffset(years=2)
    df2 = df2[df2['Admission_Date'] >= two_years_ago].copy()

    if len(df2) < 20:
        return {'error': 'Insufficient data for survival analysis'}

    # ── Overall KM curve ─────────────────────────────────────────────────
    if LIFELINES_OK:
        kmf = KaplanMeierFitter()
        kmf.fit(df2['LOS'], event_observed=df2['event'])
        overall_curve = [
            {'day': int(t), 'survival_prob': round(float(p), 4)}
            for t, p in zip(kmf.survival_function_.index, kmf.survival_function_.iloc[:, 0])
        ]
        try:
            median_los = round(float(kmf.median_survival_time_), 1)
        except Exception:
            median_los = None
    else:
        overall_curve, median_los = _km_fit_numpy(df2['LOS'].values, df2['event'].values)
        if median_los is not None:
            median_los = round(median_los, 1)

    # ── Per-bed-type KM curves ────────────────────────────────────────────
    by_bed_type = {}
    for bed in df2['Bed_Type'].value_counts().head(4).index:
        sub = df2[df2['Bed_Type'] == bed]
        if len(sub) < 5:
            continue
        try:
            if LIFELINES_OK:
                kmf_b = KaplanMeierFitter()
                kmf_b.fit(sub['LOS'], event_observed=sub['event'])
                curve = [{'day': int(t), 'survival_prob': round(float(p), 4)}
                         for t, p in zip(kmf_b.survival_function_.index,
                                         kmf_b.survival_function_.iloc[:, 0])]
                try:
                    med = round(float(kmf_b.median_survival_time_), 1)
                except Exception:
                    med = None
            else:
                curve, med = _km_fit_numpy(sub['LOS'].values, sub['event'].values)
                if med is not None:
                    med = round(float(med), 1)
            by_bed_type[str(bed)] = {'median_los': med, 'curve': curve}
        except Exception:
            pass

    # ── Cox PH regression ────────────────────────────────────────────────
    cox_result = {}
    try:
        if LIFELINES_OK:
            cox_df = df2[['LOS', 'event', 'Age']].dropna().copy()
            if 'Severity_Score' in df2.columns:
                cox_df['Severity_Score'] = pd.to_numeric(df2['Severity_Score'], errors='coerce').fillna(5)
            if 'Num_Previous_Admissions' in df2.columns:
                cox_df['Num_Previous_Admissions'] = pd.to_numeric(
                    df2['Num_Previous_Admissions'], errors='coerce').fillna(0)
            cox_df = cox_df.dropna()
            cox_df = cox_df[cox_df['LOS'] > 0]
            if len(cox_df) >= 20:
                cph = CoxPHFitter()
                cph.fit(cox_df, duration_col='LOS', event_col='event')
                sm = cph.summary
                cox_result = {
                    'summary': [
                        {'covariate': str(feat),
                         'coef':      round(float(sm.loc[feat, 'coef']), 4),
                         'exp_coef':  round(float(sm.loc[feat, 'exp(coef)']), 4),
                         'p_value':   round(float(sm.loc[feat, 'p']), 4),
                         'significant': bool(float(sm.loc[feat, 'p']) < 0.05)}
                        for feat in sm.index
                    ],
                    'concordance': round(float(cph.concordance_index_), 4),
                }
        else:
            cox_result = _cox_numpy(df2)
    except Exception as e:
        cox_result = {'error': str(e)}

    return {
        'model': 'Kaplan-Meier + Cox Proportional Hazards',
        'median_los': median_los, 'overall_curve': overall_curve,
        'by_bed_type': by_bed_type, 'cox': cox_result,
        'total_patients': int(len(df2)),
        'note': 'LOS = Length of Stay. Event=1 if discharged/transferred, 0=censored (still admitted)',
    }


# ══════════════════════════════════════════════════════════
#  MODEL 10 — predict_treatment_cost
# ══════════════════════════════════════════════════════════

def predict_treatment_cost(df):
    if 'Treatment_Cost_INR' not in df.columns:
        return {'error': 'Column Treatment_Cost_INR not found in dataset'}

    df2 = df.copy()
    for col in ['Disease', 'Bed_Type', 'Visit_Type', 'State']:
        if col in df2.columns:
            df2[col] = df2[col].astype(str).str.strip().str.title()

    df2['LOS']                = (df2['Discharge_Date'] - df2['Admission_Date']).dt.days.fillna(0).clip(0, 365)
    df2['Treatment_Cost_INR'] = pd.to_numeric(df2['Treatment_Cost_INR'], errors='coerce')
    df2 = df2.dropna(subset=['Treatment_Cost_INR', 'Age']).copy()
    df2 = df2[df2['Treatment_Cost_INR'] > 0].copy()

    if len(df2) < 50:
        return {'error': 'Insufficient data for cost prediction model'}

    feat_cols   = ['Age', 'LOS']
    feat_labels = ['Age', 'LOS']

    for col, label in [('Disease', 'Disease'), ('Bed_Type', 'Bed Type'),
                       ('Visit_Type', 'Visit Type'), ('State', 'State')]:
        if col in df2.columns:
            le = LabelEncoder()
            df2[col + '_enc'] = le.fit_transform(df2[col].astype(str))
            feat_cols.append(col + '_enc'); feat_labels.append(label)

    if 'Severity_Score' in df2.columns:
        df2['Severity_Score'] = pd.to_numeric(df2['Severity_Score'], errors='coerce').fillna(5)
        feat_cols.append('Severity_Score'); feat_labels.append('Severity Score')
    if 'SpO2' in df2.columns:
        df2['SpO2'] = pd.to_numeric(df2['SpO2'], errors='coerce').fillna(97)
        feat_cols.append('SpO2'); feat_labels.append('SpO2')
    if 'Systolic_BP' in df2.columns:
        df2['Systolic_BP'] = pd.to_numeric(df2['Systolic_BP'], errors='coerce').fillna(120)
        feat_cols.append('Systolic_BP'); feat_labels.append('Systolic BP')
    if 'Insurance_Type' in df2.columns:
        df2['Insurance_Type'] = df2['Insurance_Type'].astype(str).str.strip().str.title()
        le = LabelEncoder()
        df2['Insurance_Type_enc'] = le.fit_transform(df2['Insurance_Type'])
        feat_cols.append('Insurance_Type_enc'); feat_labels.append('Insurance Type')
    if 'Comorbidities' in df2.columns:
        df2['has_comorbidity'] = (df2['Comorbidities'].astype(str).str.strip().str.title() != 'None').astype(int)
        feat_cols.append('has_comorbidity'); feat_labels.append('Has Comorbidity')

    X = df2[feat_cols].fillna(0).values
    y = df2['Treatment_Cost_INR'].values.astype(float)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    r2_val = round(float(r2_score(y_te, y_pred)), 4)
    mae    = int(mean_absolute_error(y_te, y_pred))

    # feature_importance as [[feat, val], ...] to match app.jsx ([feat, val] destructuring)
    fi = model.feature_importances_
    feat_imp = sorted(
        [[feat_labels[i], round(float(fi[i]) * 100, 1)] for i in range(len(feat_labels))],
        key=lambda x: -x[1]
    )

    disease_cost = []
    for d in df2['Disease'].value_counts().head(10).index:
        sub = df2[df2['Disease'] == d]['Treatment_Cost_INR']
        disease_cost.append({'disease': str(d), 'avg_cost': int(sub.mean()),
                              'min_cost': int(sub.min()), 'max_cost': int(sub.max())})
    disease_cost.sort(key=lambda x: -x['avg_cost'])

    bed_type_cost = sorted(
        [{'bed_type': str(bt), 'avg_cost': int(df2[df2['Bed_Type'] == bt]['Treatment_Cost_INR'].mean())}
         for bt in df2['Bed_Type'].unique() if len(df2[df2['Bed_Type'] == bt]) > 0],
        key=lambda x: -x['avg_cost']
    )

    return {
        'model': 'Random Forest Regressor',
        'r2_score': r2_val, 'mae_inr': mae,
        'feature_importance': feat_imp,
        'disease_cost': disease_cost, 'bed_type_cost': bed_type_cost,
        'overall_avg_cost': int(df2['Treatment_Cost_INR'].mean()),
        'overall_max_cost': int(df2['Treatment_Cost_INR'].max()),
        'note': 'Cost in INR. Target: Treatment_Cost_INR column',
    }


# ══════════════════════════════════════════════════════════
#  MODEL 11 — monte_carlo_simulation
# ══════════════════════════════════════════════════════════

def monte_carlo_simulation(df, case_increase_pct=30, severe_ratio_pct=15,
                            zones=2, n_simulations=1000):
    df_clean = df.copy()
    df_clean['Admission_Status'] = df_clean['Admission_Status'].astype(str).str.strip().str.title()
    df_clean['Bed_Type']         = df_clean['Bed_Type'].astype(str).str.strip().str.title()

    admitted      = df_clean[df_clean['Admission_Status'] == 'Admitted']
    n_admitted    = int(len(admitted))

    # FIX 3: use currently occupied beds (not n_admitted) vs physical bed capacity
    total_beds    = int(df_clean['Bed_ID'].nunique())
    occupied_beds = int(admitted['Bed_ID'].nunique())

    # FIX 4: track ICU occupancy vs ICU capacity separately
    icu_bed_ids  = df_clean[df_clean['Bed_Type'] == 'Icu']['Bed_ID'].unique()
    icu_capacity = int(len(icu_bed_ids))
    icu_occupied = int(admitted[admitted['Bed_ID'].isin(icu_bed_ids)]['Bed_ID'].nunique())

    total_doctors = int(df_clean['Doctor_ID'].nunique())

    np.random.seed(42)
    case_mult   = np.random.normal(case_increase_pct / 100, 0.05, n_simulations)
    severe_mult = np.random.normal(severe_ratio_pct / 100, 0.03, n_simulations)

    # FIX 1 & 2: remove the arbitrary /4 deflator and Poisson zone_mult
    # Poisson was collapsing 13.5% of simulations to 0 (when zone draw = 0).
    # zones is now a direct multiplier: extra pressure proportional to affected zones.
    extra_cases  = np.maximum(0, n_admitted * case_mult * max(zones, 1))
    icu_needed   = np.maximum(0, extra_cases * severe_mult)
    staff_needed = np.maximum(0, extra_cases * 0.21)

    def pct_stats(arr):
        return {'p10': int(np.percentile(arr, 10)), 'p50': int(np.percentile(arr, 50)),
                'p90': int(np.percentile(arr, 90)), 'mean': int(np.mean(arr))}

    # FIX 3: surge beds needed vs free capacity (not n_admitted vs total)
    overload_prob  = round(float(np.mean((occupied_beds + extra_cases) > total_beds) * 100), 1)
    # FIX 4: surge ICU needed vs free ICU beds (not just icu_needed > icu_capacity)
    icu_short_prob = round(float(np.mean((icu_occupied + icu_needed) > max(icu_capacity, 1)) * 100), 1)

    hist_counts, hist_edges = np.histogram(extra_cases, bins=20)

    ep_stats                   = pct_stats(extra_cases)
    icu_stats                  = pct_stats(icu_needed)
    icu_stats['capacity']      = icu_capacity
    icu_stats['occupied']      = icu_occupied
    icu_stats['available']     = icu_capacity - icu_occupied
    icu_stats['shortage_prob'] = icu_short_prob
    staff_stats                = pct_stats(staff_needed)
    staff_stats['available']   = total_doctors

    # histogram as [{bin, count}, ...] so the frontend BarChart can use dataKey="bin"/"count"
    histogram = [
        {'bin': int(hist_edges[i]), 'count': int(hist_counts[i])}
        for i in range(len(hist_counts))
    ]

    return {
        'model': 'Monte Carlo Simulation (1000 runs)',
        'inputs': {
            'current_admitted':  n_admitted,
            'occupied_beds':     occupied_beds,
            'total_beds':        total_beds,
            'icu_occupied':      icu_occupied,
            'icu_capacity':      icu_capacity,
            'case_increase_pct': case_increase_pct,
            'severe_ratio_pct':  severe_ratio_pct,
            'zones':             zones,
            'n_simulations':     n_simulations,
        },
        'extra_patients':       ep_stats,
        'icu_beds_needed':      icu_stats,
        'extra_staff':          staff_stats,
        'system_overload_prob': overload_prob,
        'histogram':            histogram,
    }


# ══════════════════════════════════════════════════════════
#  MODEL 12 — symptom_disease_classifier
# ══════════════════════════════════════════════════════════

def symptom_disease_classifier(df):
    if 'Symptoms' not in df.columns:
        return {'error': 'Column Symptoms not found in dataset'}

    df2 = df.copy()
    df2['Symptoms'] = df2['Symptoms'].astype(str).str.lower().str.strip()
    df2['Disease']  = df2['Disease'].astype(str).str.strip().str.title()
    df2 = df2.dropna(subset=['Symptoms', 'Disease']).copy()
    df2 = df2[df2['Symptoms'] != 'nan'].copy()

    valid = df2['Disease'].value_counts()
    df2   = df2[df2['Disease'].isin(valid[valid >= 5].index)].copy()

    if len(df2) < 50 or df2['Disease'].nunique() < 3:
        return {'error': 'Need ≥ 50 rows and ≥ 3 disease classes with ≥ 5 samples each'}

    X = df2['Symptoms'].values
    y = df2['Disease'].values
    strat = y if len(np.unique(y)) > 1 else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=1000,
                                  stop_words='english', sublinear_tf=True)),
        ('clf',   LogisticRegression(max_iter=1000, C=1.0, random_state=42,
                                     class_weight='balanced')),
    ])
    pipeline.fit(X_tr, y_tr)
    accuracy = round(float(pipeline.score(X_te, y_te)) * 100, 1)
    try:
        cv_accuracy = round(float(cross_val_score(pipeline, X, y, cv=5, scoring='accuracy').mean()) * 100, 1)
    except Exception:
        cv_accuracy = accuracy

    tfidf      = pipeline.named_steps['tfidf']
    clf        = pipeline.named_steps['clf']
    feat_names = tfidf.get_feature_names_out()
    classes    = clf.classes_

    top_terms = {}
    for i, cls in enumerate(classes):
        try:
            top_idx = clf.coef_[i].argsort()[-5:][::-1]
            top_terms[str(cls)] = [str(feat_names[j]) for j in top_idx]
        except Exception:
            top_terms[str(cls)] = []

    examples = [
        "high fever chills sweating headache nausea",
        "chest pain shortness of breath palpitations",
        "persistent cough blood sputum night sweats",
        "red eyes discharge itching light sensitivity",
        "severe headache stiff neck confusion fever",
    ]
    example_preds = []
    for ex in examples:
        try:
            proba     = pipeline.predict_proba([ex])[0]
            pred_idx  = int(np.argmax(proba))
            top3_idx  = proba.argsort()[-3:][::-1]
            example_preds.append({
                'input_symptoms':    ex,
                'predicted_disease': str(classes[pred_idx]),
                'confidence':        round(float(proba[pred_idx]) * 100, 1),
                'top3_predictions':  [{'disease': str(classes[j]), 'prob': round(float(proba[j]) * 100, 1)}
                                      for j in top3_idx],
            })
        except Exception:
            pass

    n_classes = int(df2['Disease'].nunique())
    return {
        'model': 'TF-IDF (1-2 gram) + Logistic Regression',
        'accuracy_pct': accuracy, 'cv_accuracy_pct': cv_accuracy,
        'num_diseases': n_classes,
        'top_terms_per_disease': top_terms,
        'example_predictions': example_preds,
        'note': f'Trained on Symptoms column. {n_classes} disease classes.',
    }