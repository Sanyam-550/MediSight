"""
db_connector.py — MediSight v4
Sole data source: data/patients.csv

Cleans all 34 columns including:
  - Symptoms, Comorbidities, Outcome, Insurance_Type
  - Readmitted_Within_30_Days
"""

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

CSV_PATH = "data/patients.csv"

_df_cache = None


def _get_df():
    global _df_cache
    if _df_cache is None:
        load_data()
    return _df_cache


def _clean_df(df):
    """Clean and normalize all 34 columns."""
    df['Admission_Date'] = pd.to_datetime(df['Admission_Date'], dayfirst=True, errors='coerce')
    df['Discharge_Date'] = pd.to_datetime(df['Discharge_Date'], dayfirst=True, errors='coerce')

    text_cols = [
        'Disease', 'State', 'City', 'Bed_Type', 'Admission_Status', 'Visit_Type',
        'Symptoms', 'Comorbidities', 'Outcome', 'Insurance_Type',
        'Doctor_Specialization', 'First_Name', 'Last_Name', 'Gender', 'Blood_Group'
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    if 'Readmitted_Within_30_Days' in df.columns:
        df['Readmitted_Within_30_Days'] = df['Readmitted_Within_30_Days'].astype(str).str.strip().str.title()

    return df


def load_data():
    """Load and cache the CSV dataset."""
    global _df_cache
    _df_cache = _clean_df(pd.read_csv(CSV_PATH))
    print(f"[DB] Loaded {len(_df_cache)} rows from CSV '{CSV_PATH}'")
    return _df_cache


def get_source():
    return "CSV"


# ─────────────────────────────────────────────
# sql_* functions — pandas queries on the CSV
# DataFrame, same return schema as before.
# ─────────────────────────────────────────────

def sql_disease_summary():
    df = _get_df()
    result = df.groupby('Disease', as_index=False).agg(
        total_patients=('Disease', 'count'),
        currently_admitted=('Admission_Status', lambda x: (x == 'Admitted').sum()),
        discharged=('Admission_Status', lambda x: (x == 'Discharged').sum()),
    )
    return result.sort_values('currently_admitted', ascending=False).head(15).to_dict(orient='records')


def sql_state_burden():
    df = _get_df()
    result = df.groupby('State', as_index=False).agg(
        total_patients=('Patient_ID', 'nunique'),
        beds_used=('Bed_ID', 'nunique'),
        doctors=('Doctor_ID', 'nunique'),
        admitted=('Admission_Status', lambda x: (x == 'Admitted').sum()),
    )
    return result.sort_values('admitted', ascending=False).to_dict(orient='records')


def sql_doctor_workload():
    df = _get_df()
    admitted = df[df['Admission_Status'] == 'Admitted']
    result = admitted.groupby('Doctor_Specialization', as_index=False).agg(
        doctors=('Doctor_ID', 'nunique'),
        patients=('Patient_ID', 'nunique'),
        beds_used=('Bed_ID', 'nunique'),
    )
    return result.sort_values('patients', ascending=False).to_dict(orient='records')


def sql_daily_admissions_last_n(days=30):
    df = _get_df()
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=int(days))
    filtered = df[df['Admission_Date'] >= cutoff]
    result = (
        filtered.groupby(filtered['Admission_Date'].dt.date)
        .size()
        .reset_index(name='count')
        .rename(columns={'Admission_Date': 'date'})
        .sort_values('date')
    )
    result['date'] = result['date'].astype(str)
    return result.to_dict(orient='records')


def sql_bed_type_utilisation():
    df = _get_df()
    result = df.groupby('Bed_Type', as_index=False).agg(
        total_beds=('Bed_ID', 'nunique'),
        occupied=('Admission_Status', lambda x: (x == 'Admitted').sum()),
    )
    result = result.sort_values('occupied', ascending=False)
    result['occupancy_pct'] = (result['occupied'] / result['total_beds'] * 100).round(1)
    return result.to_dict(orient='records')


def sql_blood_group_distribution():
    df = _get_df()
    result = (
        df.groupby('Blood_Group', as_index=False)['Patient_ID']
        .nunique()
        .rename(columns={'Patient_ID': 'patients'})
        .sort_values('patients', ascending=False)
    )
    return result.to_dict(orient='records')


def sql_age_distribution():
    df = _get_df()
    bins   = [0, 17, 34, 49, 64, float('inf')]
    labels = ['Under 18', '18-34', '35-49', '50-64', '65+']
    temp = df.copy()
    temp['age_band'] = pd.cut(temp['Age'], bins=bins, labels=labels, right=True)
    result = (
        temp.groupby('age_band', as_index=False, observed=True)['Patient_ID']
        .nunique()
        .rename(columns={'Patient_ID': 'patients'})
    )
    return result.to_dict(orient='records')


def sql_export_filtered(state=None, disease=None, status=None, limit=500):
    df = _get_df()
    mask = pd.Series(True, index=df.index)
    if state:   mask &= df['State']   == state.title()
    if disease: mask &= df['Disease'] == disease.title()
    if status:  mask &= df['Admission_Status'] == status.title()

    cols = [
        'Patient_ID', 'First_Name', 'Last_Name', 'Age', 'Gender', 'Disease',
        'State', 'City', 'Admission_Status', 'Bed_Type', 'Doctor_Name',
        'Doctor_Specialization', 'Admission_Date', 'Discharge_Date',
    ]
    available = [c for c in cols if c in df.columns]
    result = df[mask][available].head(int(limit)).copy()
    result['Admission_Date'] = result['Admission_Date'].astype(str)
    result['Discharge_Date'] = result['Discharge_Date'].astype(str)
    return result.to_dict(orient='records')
