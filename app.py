"""
app.py — MediSight v4
Data source: CSV (data/patients.csv)
All ML models and dashboard endpoints operate on the same DataFrame
loaded once at startup via db_connector.load_data()

BUG FIXED:
  BUG 5 (app.py): ML endpoints had NO exception handling.
  The SQL endpoints all used _sql_safe() but every ML route was bare:
      @app.route('/api/ml/disease-risk')
      def api_ml_disease_risk():
          return jsonify(train_disease_risk_model(df))   ← uncaught exception = raw 500 HTML
  When any ML function throws (e.g. the sklearn bugs fixed in ml_model.py, or
  a data issue), Flask returns a raw HTML error page. The frontend's apiFetch()
  calls .json() on it, which fails → returns null → every model shows "unavailable".
  FIX: added _ml_safe() wrapper identical in pattern to _sql_safe(), so exceptions
  are caught and returned as {"error": "...", "hint": "..."} JSON instead.
"""

import uuid
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# ── Data layer (CSV)
from db_connector import (
    load_data, get_source,
    sql_disease_summary, sql_state_burden, sql_doctor_workload,
    sql_daily_admissions_last_n, sql_bed_type_utilisation,
    sql_blood_group_distribution, sql_age_distribution, sql_export_filtered,
)

# ── Analytics / processing
from data_processor import (
    get_kpis, get_resource_utilisation, get_alert_feed,
    get_capacity_by_city, get_hospital_table, get_daily_admissions,
    get_disease_trends, get_outbreak_cards, get_state_heatmap,
    get_resource_by_department,
)

# ── ML models
from ml_model import (
    train_and_forecast,
    train_disease_risk_model,
    detect_admission_anomalies,
    cluster_patients,
    forecast_bed_demand_by_disease,
    forecast_doctor_workload,
    outbreak_early_warning,
    predict_readmission,
    survival_analysis,
    predict_treatment_cost,
    monte_carlo_simulation,
    symptom_disease_classifier,
)

app = Flask(__name__, static_folder='static')
CORS(app)

# ── In-memory auth store ─────────────────────────────────────────────────────
# Sample demo account shown on the login page
USERS = {
    'admin@medisight.com': {
        'password': 'MediSight@123',
        'name':     'Dr. Admin',
        'doctor_id': 'DOC-00001',
    }
}
ACTIVE_TOKENS = {}   # token (str uuid) → email


# ── Auth endpoints ────────────────────────────────────────────────────────────
@app.route('/api/auth/login', methods=['POST'])
def api_auth_login():
    data     = request.get_json(silent=True) or {}
    email    = data.get('email', '').lower().strip()
    password = data.get('password', '')
    user     = USERS.get(email)
    if not user or user['password'] != password:
        return jsonify({'error': 'Invalid email or password'}), 401
    token = str(uuid.uuid4())
    ACTIVE_TOKENS[token] = email
    return jsonify({'token': token, 'name': user['name'], 'email': email})


@app.route('/api/auth/signup', methods=['POST'])
def api_auth_signup():
    data      = request.get_json(silent=True) or {}
    email     = data.get('email', '').lower().strip()
    password  = data.get('password', '')
    name      = data.get('name', '').strip()
    username  = data.get('username', '').strip()
    doctor_id = data.get('doctorId', '').strip()
    if not email or not password or not name:
        return jsonify({'error': 'Name, email, and password are required'}), 400
    if email in USERS:
        return jsonify({'error': 'An account with this email already exists'}), 409
    if len(password) < 8:
        return jsonify({'error': 'Password must be at least 8 characters'}), 400
    USERS[email] = {'password': password, 'name': name,
                    'doctor_id': doctor_id, 'username': username}
    return jsonify({'message': 'Account created successfully'})


@app.route('/api/auth/verify')
def api_auth_verify():
    token = request.headers.get('Authorization', '').replace('Bearer ', '').strip()
    if token and token in ACTIVE_TOKENS:
        email = ACTIVE_TOKENS[token]
        user  = USERS.get(email, {})
        return jsonify({'valid': True, 'email': email, 'name': user.get('name', '')})
    return jsonify({'valid': False}), 401


@app.route('/api/auth/logout', methods=['POST'])
def api_auth_logout():
    token = request.headers.get('Authorization', '').replace('Bearer ', '').strip()
    ACTIVE_TOKENS.pop(token, None)
    return jsonify({'message': 'Logged out'})


@app.route("/login")
def login():
    return send_from_directory("static", "login.html")

@app.route("/signup")
def signup():
    return send_from_directory("static", "signup.html")

@app.route("/forgot-token")
def forgot_token():
    return send_from_directory("static", "forgot-token.html")

@app.route("/terms")
def terms():
    return send_from_directory("static", "terms.html")

@app.route("/support")
def support():
    return send_from_directory("static", "support.html")

@app.route("/privacy")
def privacy():
    return send_from_directory("static", "privacy.html")
print("=" * 50)
print("MediSight v4 — Starting up")
df = load_data()

# ── Frontend — login is the entry point; dashboard requires auth token in JS
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')  # auth guard in index.html JS


# ── Data source info
@app.route('/api/source')
def api_source():
    return jsonify({'source': get_source(), 'rows': len(df)})


# ═══════════════════════════════════════
#  SHARED SAFE WRAPPERS
# ═══════════════════════════════════════

def _sql_safe(fn, *args, **kwargs):
    """Wraps SQL-direct calls — returns JSON error instead of 500 HTML on failure."""
    try:
        return jsonify(fn(*args, **kwargs))
    except Exception as e:
        return jsonify({'error': str(e), 'hint': 'Check data/patients.csv'}), 500


# ── FIX BUG 5 ─────────────────────────────────────────────────────────────────
# BEFORE (broken): every ML endpoint was completely bare:
#     return jsonify(train_disease_risk_model(df))
#     Any unhandled Python exception → Flask returns raw HTML 500 page
#     → frontend apiFetch() calls .json() on HTML → throws → returns null
#     → frontend shows "Model unavailable" for every single ML model
#
# AFTER (fixed): _ml_safe() catches exceptions and returns proper JSON,
#     so the frontend always gets parseable JSON and can show a real error message.
# ──────────────────────────────────────────────────────────────────────────────
def _ml_safe(fn, *args, **kwargs):
    """Wraps ML model calls — returns JSON error instead of 500 HTML on failure."""
    try:
        return jsonify(fn(*args, **kwargs))
    except Exception as e:
        return jsonify({
            'error': str(e),
            'hint':  'Check ml_model.py or your dataset for this model',
        }), 500


# ═══════════════════════════════════════
#  DASHBOARD
# ═══════════════════════════════════════
@app.route('/api/dashboard/kpis')
def api_kpis():
    return jsonify(get_kpis(df))

@app.route('/api/dashboard/resources')
def api_resources():
    return jsonify(get_resource_utilisation(df))

@app.route('/api/dashboard/alerts')
def api_alerts():
    return jsonify(get_alert_feed(df))


# ═══════════════════════════════════════
#  CAPACITY
# ═══════════════════════════════════════
@app.route('/api/capacity/by-city')
def api_capacity_city():
    return jsonify(get_capacity_by_city(df))

@app.route('/api/capacity/hospitals')
def api_hospitals():
    return jsonify(get_hospital_table(df))


# ═══════════════════════════════════════
#  PREDICTION
# ═══════════════════════════════════════
@app.route('/api/prediction/forecast')
def api_forecast():
    days = int(request.args.get('days', 14))
    return _ml_safe(train_and_forecast, df, forecast_days=days)   # FIX BUG 5

@app.route('/api/prediction/daily')
def api_daily():
    return jsonify(get_daily_admissions(df))


# ═══════════════════════════════════════
#  OUTBREAK
# ═══════════════════════════════════════
@app.route('/api/outbreak/cards')
def api_outbreak_cards():
    return jsonify(get_outbreak_cards(df))

@app.route('/api/outbreak/trends')
def api_outbreak_trends():
    return jsonify(get_disease_trends(df))


# ═══════════════════════════════════════
#  HEATMAP
# ═══════════════════════════════════════
@app.route('/api/heatmap/states')
def api_heatmap():
    return jsonify(get_state_heatmap(df))


# ═══════════════════════════════════════
#  RESOURCES
# ═══════════════════════════════════════
@app.route('/api/resources/by-department')
def api_resources_dept():
    return jsonify(get_resource_by_department(df))

@app.route('/api/resources/utilisation')
def api_resources_util():
    return jsonify(get_resource_utilisation(df))


# ═══════════════════════════════════════
#  SCENARIO
# ═══════════════════════════════════════
@app.route('/api/scenario/impact')
def api_scenario():
    # The scenario section expects plain numbers (not Monte Carlo percentile dicts).
    # We compute a deterministic estimate here; /api/ml/monte-carlo handles the
    # full probabilistic simulation for the ML Intelligence panel.
    try:
        case_pct   = float(request.args.get('case_increase', 30))
        severe_pct = float(request.args.get('severe_ratio', 15))
        zones      = int(request.args.get('zones', 2))
        admitted   = df[df['Admission_Status'].astype(str).str.strip().str.title() == 'Admitted']
        current    = int(len(admitted))
        extra      = int(current * (case_pct / 100) * max(zones, 1))
        return jsonify({
            'current_admitted': current,
            'extra_patients':   extra,
            'icu_beds_needed':  int(extra * (severe_pct / 100)),
            'extra_staff':      int(extra * 0.21),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════
#  ML ENDPOINTS  ← ALL NOW USE _ml_safe()
# ═══════════════════════════════════════
@app.route('/api/ml/disease-risk')
def api_ml_disease_risk():
    return _ml_safe(train_disease_risk_model, df)                  # FIX BUG 5

@app.route('/api/ml/anomalies')
def api_ml_anomalies():
    return _ml_safe(detect_admission_anomalies, df)                # FIX BUG 5

@app.route('/api/ml/clusters')
def api_ml_clusters():
    k = int(request.args.get('k', 4))
    return _ml_safe(cluster_patients, df, n_clusters=k)            # FIX BUG 5

@app.route('/api/ml/bed-demand')
def api_ml_bed_demand():
    days  = int(request.args.get('days', 14))
    top_n = int(request.args.get('top_n', 5))
    return _ml_safe(forecast_bed_demand_by_disease, df,            # FIX BUG 5
                    forecast_days=days, top_n=top_n)

@app.route('/api/ml/workload')
def api_ml_workload():
    days = int(request.args.get('days', 14))
    return _ml_safe(forecast_doctor_workload, df, forecast_days=days)  # FIX BUG 5

@app.route('/api/ml/outbreak-warning')
def api_ml_outbreak_warning():
    window      = int(request.args.get('window', 7))
    z_threshold = float(request.args.get('z_threshold', 2.0))
    return _ml_safe(outbreak_early_warning, df,                    # FIX BUG 5
                    window=window, z_threshold=z_threshold)

@app.route('/api/ml/readmission')
def api_ml_readmission():
    return _ml_safe(predict_readmission, df)

@app.route('/api/ml/survival')
def api_ml_survival():
    return _ml_safe(survival_analysis, df)

@app.route('/api/ml/cost-prediction')
def api_ml_cost():
    return _ml_safe(predict_treatment_cost, df)

@app.route('/api/ml/monte-carlo')
def api_ml_monte_carlo():
    case_pct   = float(request.args.get('case_increase', 30))
    severe_pct = float(request.args.get('severe_ratio', 15))
    zones      = int(request.args.get('zones', 2))
    n_sim      = int(request.args.get('simulations', 1000))
    return _ml_safe(monte_carlo_simulation, df,
                    case_increase_pct=case_pct,
                    severe_ratio_pct=severe_pct,
                    zones=zones, n_simulations=n_sim)

@app.route('/api/ml/symptom-classifier')
def api_ml_symptoms():
    return _ml_safe(symptom_disease_classifier, df)


# ═══════════════════════════════════════
#  SQL ENDPOINTS (pandas queries on CSV)
# ═══════════════════════════════════════
@app.route('/api/sql/status')
def api_sql_status():
    return jsonify({'status': 'ok', 'source': 'CSV', 'rows': len(df)})

@app.route('/api/sql/disease-summary')
def api_sql_disease():
    return _sql_safe(sql_disease_summary)

@app.route('/api/sql/state-burden')
def api_sql_state():
    return _sql_safe(sql_state_burden)

@app.route('/api/sql/doctor-workload')
def api_sql_doctor():
    return _sql_safe(sql_doctor_workload)

@app.route('/api/sql/daily')
def api_sql_daily():
    return _sql_safe(sql_daily_admissions_last_n, int(request.args.get('days', 30)))

@app.route('/api/sql/bed-utilisation')
def api_sql_beds():
    return _sql_safe(sql_bed_type_utilisation)

@app.route('/api/sql/blood-groups')
def api_sql_blood():
    return _sql_safe(sql_blood_group_distribution)

@app.route('/api/sql/age-distribution')
def api_sql_age():
    return _sql_safe(sql_age_distribution)

@app.route('/api/sql/export')
def api_sql_export():
    return _sql_safe(
        sql_export_filtered,
        request.args.get('state'),
        request.args.get('disease'),
        request.args.get('status'),
        int(request.args.get('limit', 500)),
    )


# ═══════════════════════════════════════
#  RUN
# ═══════════════════════════════════════
if __name__ == '__main__':
    app.run(debug=True, port=5000)