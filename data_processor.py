import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def get_kpis(df):
    total_admitted   = len(df[df['Admission_Status'] == 'Admitted'])
    total_discharged = len(df[df['Admission_Status'] == 'Discharged'])

    total_beds    = df['Bed_ID'].nunique()
    occupied_beds = len(df[df['Admission_Status'] == 'Admitted']['Bed_ID'].dropna().unique())
    occupancy_pct = round((occupied_beds / total_beds * 100), 1) if total_beds > 0 else 0

    doctors_on_duty = df['Doctor_ID'].nunique()

    disease_counts = (
        df[df['Admission_Status'] == 'Admitted']
        .groupby('Disease')
        .size()
        .reset_index(name='count')
    )
    active_alerts = len(disease_counts[disease_counts['count'] >= 10])

    return {
        'total_admitted':    int(total_admitted),
        'total_discharged':  int(total_discharged),
        'bed_occupancy_pct': occupancy_pct,
        'doctors_on_duty':   int(doctors_on_duty),
        'active_alerts':     int(active_alerts),
    }


def get_resource_utilisation(df):
    admitted  = df[df['Admission_Status'] == 'Admitted']
    bed_types = df.groupby('Bed_Type')['Bed_ID'].nunique().to_dict()
    used_beds = admitted.groupby('Bed_Type')['Bed_ID'].nunique().to_dict()

    result = []
    for btype, total in bed_types.items():
        used = used_beds.get(btype, 0)
        pct  = round(used / total * 100, 1) if total > 0 else 0
        result.append({'bed_type': btype, 'total': int(total), 'used': int(used), 'pct': pct})

    return sorted(result, key=lambda x: -x['pct'])


def get_alert_feed(df):
    admitted = df[df['Admission_Status'] == 'Admitted']
    counts   = admitted.groupby('Disease').size().reset_index(name='count')
    counts   = counts.sort_values('count', ascending=False).head(8)

    alerts = []
    for _, row in counts.iterrows():
        count    = int(row['count'])
        severity = 'critical' if count >= 30 else 'warning' if count >= 15 else 'watch'
        states   = admitted[admitted['Disease'] == row['Disease']]['State'].value_counts()
        top_state = str(states.index[0]) if len(states) > 0 else 'Multiple'
        alerts.append({'disease': row['Disease'], 'count': count,
                        'severity': severity, 'region': top_state})
    return alerts


def get_capacity_by_city(df):
    admitted      = df[df['Admission_Status'] == 'Admitted']
    total_by_city = df.groupby('City')['Bed_ID'].nunique()
    used_by_city  = admitted.groupby('City')['Bed_ID'].nunique()

    result = []
    for city in total_by_city.index:
        total = int(total_by_city[city])
        used  = int(used_by_city.get(city, 0))
        pct   = round(used / total * 100, 1) if total > 0 else 0
        result.append({'city': city, 'total': total, 'used': used, 'pct': pct})

    return sorted(result, key=lambda x: -x['pct'])[:10]


def get_hospital_table(df):
    admitted = df[df['Admission_Status'] == 'Admitted']

    summary = df.groupby(['City', 'State']).agg(
        total_beds     = ('Bed_ID',      'nunique'),
        total_patients = ('Patient_ID',  'nunique'),
        doctors        = ('Doctor_ID',   'nunique'),
    ).reset_index()

    admitted_counts = admitted.groupby(['City', 'State']).size().reset_index(name='admitted')
    summary = summary.merge(admitted_counts, on=['City', 'State'], how='left').fillna(0)
    summary['admitted']      = summary['admitted'].astype(int)
    summary['occupancy_pct'] = (
        (summary['admitted'] / summary['total_beds'] * 100).clip(0, 100).round(1)
    )

    def status(pct):
        if pct >= 85: return 'Critical'
        if pct >= 70: return 'Warning'
        return 'Normal'

    summary['status'] = summary['occupancy_pct'].apply(status)
    return summary.sort_values('occupancy_pct', ascending=False).head(10).to_dict(orient='records')


def get_daily_admissions(df):
    daily = (
        df.dropna(subset=['Admission_Date'])
        .groupby(df['Admission_Date'].dt.date)
        .size()
        .reset_index(name='count')
        .rename(columns={'Admission_Date': 'date'})
        .sort_values('date')
    )
    daily['date'] = daily['date'].astype(str)
    return daily.to_dict(orient='records')


def get_disease_trends(df):
    """Return monthly admission counts for the top 5 diseases.
    Shape: {'labels': ['2024-01', ...], 'diseases': {'Malaria': [10, 12, ...], ...}}
    """
    df2 = df.dropna(subset=['Admission_Date']).copy()
    df2['month'] = df2['Admission_Date'].dt.to_period('M').astype(str)

    top_diseases = df2['Disease'].value_counts().head(5).index.tolist()
    trend  = df2[df2['Disease'].isin(top_diseases)]
    pivot  = (
        trend.groupby(['month', 'Disease'])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    labels   = pivot.index.tolist()
    diseases = {str(col): [int(v) for v in pivot[col].tolist()] for col in pivot.columns}
    return {'labels': labels, 'diseases': diseases}


def get_outbreak_cards(df):
    """Top-6 diseases by admitted count with risk score and level."""
    admitted      = df[df['Admission_Status'] == 'Admitted']
    disease_counts = (
        admitted.groupby('Disease').size()
        .reset_index(name='total')
        .sort_values('total', ascending=False)
        .head(6)
    )

    max_count = int(disease_counts['total'].max()) if len(disease_counts) > 0 else 1

    result = []
    for _, row in disease_counts.iterrows():
        total      = int(row['total'])
        risk_score = round(min(total / max_count * 10, 10), 1)
        level      = 'critical' if risk_score >= 7 else 'warning' if risk_score >= 4 else 'watch'

        top_state_s = admitted[admitted['Disease'] == row['Disease']]['State'].value_counts()
        top_state   = str(top_state_s.index[0]) if len(top_state_s) > 0 else 'Multiple'

        result.append({
            'disease':    str(row['Disease']),
            'total':      total,
            'top_state':  top_state,
            'risk_score': risk_score,
            'level':      level,
        })
    return result


def get_state_heatmap(df):
    """Per-state patient burden, risk score (0-100), and top disease."""
    state_agg = df.groupby('State').agg(
        total    = ('Patient_ID',      'nunique'),
        admitted = ('Admission_Status', lambda x: (x == 'Admitted').sum()),
    ).reset_index()

    max_admitted = int(state_agg['admitted'].max()) if len(state_agg) > 0 else 1

    result = []
    for _, row in state_agg.iterrows():
        state    = str(row['State'])
        total    = int(row['total'])
        admitted = int(row['admitted'])
        risk_score = int(min(admitted / max(max_admitted, 1) * 100, 100))
        risk_level = 'critical' if risk_score >= 70 else 'warning' if risk_score >= 40 else 'normal'

        top_dis_s  = df[df['State'] == state]['Disease'].value_counts()
        top_disease = str(top_dis_s.index[0]) if len(top_dis_s) > 0 else '—'

        result.append({
            'State':       state,
            'total':       total,
            'admitted':    admitted,
            'risk_score':  risk_score,
            'risk_level':  risk_level,
            'top_disease': top_disease,
        })

    result.sort(key=lambda x: -x['risk_score'])
    return result


def get_resource_by_department(df):
    """Admitted patient count, doctor count, and beds used per specialization."""
    admitted = df[df['Admission_Status'] == 'Admitted']

    if 'Doctor_Specialization' not in df.columns:
        return []

    dept = admitted.groupby('Doctor_Specialization').agg(
        patients  = ('Patient_ID', 'nunique'),
        doctors   = ('Doctor_ID',  'nunique'),
        beds_used = ('Bed_ID',     'nunique'),
    ).reset_index()

    return dept.sort_values('patients', ascending=False).to_dict(orient='records')
