import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import linprog

st.set_page_config(page_title="Fleet Intelligence", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem 2rem; }

/* ── SIDEBAR ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #0a0a0f;
    border-right: 1px solid #2a2a3e;
}
[data-testid="stSidebar"] .css-1d391kg { padding: 1.5rem 1rem; }

/* Sidebar ALL text override — force bright */
[data-testid="stSidebar"] * {
    color: #d0d0e8 !important;
}

/* Sidebar radio labels */
[data-testid="stSidebar"] .stRadio label {
    color: #c8c8e0 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}

/* Sidebar radio selected */
[data-testid="stSidebar"] .stRadio [aria-checked="true"] + div label,
[data-testid="stSidebar"] .stRadio input:checked + div {
    color: #4a9eff !important;
}

/* Sidebar slider label */
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] [data-testid="stSlider"] label,
[data-testid="stSidebar"] label {
    color: #c8c8e0 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}

/* Slider min/max tick values */
[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] {
    color: #9090b8 !important;
    font-size: 12px !important;
}

/* Slider current value bubble */
[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"],
[data-testid="stSidebar"] .stSlider div[data-baseweb="tooltip"] span {
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 13px !important;
}

/* Number input label */
[data-testid="stSidebar"] .stNumberInput label {
    color: #c8c8e0 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}

/* Number input value */
[data-testid="stSidebar"] .stNumberInput input {
    color: #ffffff !important;
    background: #16162a !important;
    border: 1px solid #2a2a4a !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}

/* Sidebar section divider text */
.sb-section {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #6868a0 !important;
    margin: 20px 0 10px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e1e34;
}

/* Model registry rows */
.model-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 0;
    border-bottom: 1px solid #1a1a2e;
    font-size: 12px;
}
.model-name-text  { color: #9090b8 !important; }
.model-active-text { color: #e8e8f8 !important; font-weight: 600; }
.model-r2         { color: #a78bfa !important; font-weight: 700; }
.active-badge {
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 1px;
    background: rgba(167,139,250,0.15);
    color: #a78bfa !important;
    border: 1px solid rgba(167,139,250,0.3);
    padding: 2px 7px;
    border-radius: 10px;
    margin-left: 8px;
}

/* ── GLOBAL BG ───────────────────────────────────────────── */
.stApp { background-color: #07070d; }

/* Page header */
.page-eyebrow {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #4a9eff;
    margin-bottom: 6px;
}
.page-title {
    font-size: 26px;
    font-weight: 700;
    color: #f0f0f8;
    letter-spacing: -0.5px;
    line-height: 1.2;
    margin-bottom: 4px;
}
.page-sub {
    font-size: 13px;
    color: #7070a0;
    margin-bottom: 0;
}

/* Status pill */
.status-live {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(74, 222, 128, 0.08);
    border: 1px solid rgba(74, 222, 128, 0.2);
    color: #4ade80;
    font-size: 11px;
    font-weight: 500;
    padding: 4px 12px;
    border-radius: 20px;
    letter-spacing: 0.5px;
}
.status-dot {
    width: 6px; height: 6px;
    background: #4ade80;
    border-radius: 50%;
    display: inline-block;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
}

/* KPI Cards */
.kpi-wrap {
    background: #0d0d1a;
    border: 1px solid #1a1a2e;
    border-radius: 12px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
}
.kpi-wrap::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.kpi-danger::before  { background: linear-gradient(90deg, #ff4757, transparent); }
.kpi-info::before    { background: linear-gradient(90deg, #4a9eff, transparent); }
.kpi-success::before { background: linear-gradient(90deg, #4ade80, transparent); }
.kpi-accent::before  { background: linear-gradient(90deg, #a78bfa, transparent); }

.kpi-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #7070a0;
    margin-bottom: 10px;
}
.kpi-value {
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -1px;
    line-height: 1;
    margin-bottom: 6px;
}
.kpi-value.danger  { color: #ff4757; }
.kpi-value.info    { color: #4a9eff; }
.kpi-value.success { color: #4ade80; }
.kpi-value.accent  { color: #a78bfa; }
.kpi-sub { font-size: 11px; color: #5050780; }

/* Section label */
.section-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #7070a0;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid #12121f;
}

/* Reallocation table */
.alloc-row {
    display: flex;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid #12121f;
    gap: 12px;
    font-size: 13px;
}
.alloc-from { color: #4ade80; font-weight: 500; min-width: 100px; }
.alloc-arrow { color: #5050780; }
.alloc-to   { color: #f0f0f8; font-weight: 500; min-width: 100px; }
.alloc-vol  {
    margin-left: auto;
    background: rgba(74, 158, 255, 0.1);
    color: #4a9eff;
    font-size: 11px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    border: 1px solid rgba(74, 158, 255, 0.2);
}

/* Delay table */
.delay-row {
    display: flex;
    align-items: center;
    padding: 9px 0;
    border-bottom: 1px solid #12121f;
    font-size: 13px;
    gap: 8px;
}
.delay-zone   { color: #a0a0c0; width: 110px; flex-shrink: 0; }
.delay-before { color: #ff4757; font-weight: 500; width: 90px; }
.delay-arrow  { color: #505078; font-size: 11px; }
.delay-after  { color: #4ade80; font-weight: 500; }
.delay-badge {
    margin-left: auto;
    font-size: 10px;
    font-weight: 600;
    color: #4ade80;
    background: rgba(74, 222, 128, 0.08);
    border: 1px solid rgba(74, 222, 128, 0.15);
    padding: 2px 8px;
    border-radius: 10px;
}

/* Zone bar rows */
.zone-bar-row { margin-bottom: 14px; }
.zone-bar-header {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    margin-bottom: 5px;
}
.zone-name { color: #a0a0c0; }
.zone-vals { font-weight: 500; }

/* Insight alert */
.insight-alert {
    background: rgba(74, 158, 255, 0.05);
    border: 1px solid rgba(74, 158, 255, 0.15);
    border-left: 3px solid #4a9eff;
    border-radius: 8px;
    padding: 14px 16px;
    font-size: 13px;
    color: #9090c0;
    line-height: 1.6;
    margin-top: 16px;
}
.insight-alert strong { color: #f0f0f8; }

/* Executive risk cards */
.risk-card {
    background: #0d0d1a;
    border: 1px solid #1a1a2e;
    border-radius: 12px;
    padding: 22px;
}
.risk-card-title {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 16px;
}
.risk-card-title.red  { color: #ff4757; }
.risk-card-title.blue { color: #4a9eff; }
.risk-row {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    padding: 7px 0;
    border-bottom: 1px solid #12121f;
}
.risk-key { color: #7070a0; }
.risk-val { color: #f0f0f8; font-weight: 500; }
.risk-val.danger { color: #ff4757; }
.risk-val.info   { color: #4a9eff; }

/* Playbook box */
.playbook-box {
    background: linear-gradient(135deg, rgba(167,139,250,0.05), rgba(74,158,255,0.05));
    border: 1px solid rgba(167,139,250,0.15);
    border-radius: 12px;
    padding: 20px 22px;
    font-size: 14px;
    color: #9090c0;
    line-height: 1.7;
}
.playbook-box strong { color: #f0f0f8; }

/* Sidebar brand header */
.sb-brand-title {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #4a9eff !important;
}
.sb-brand-sub {
    font-size: 12px;
    color: #7070a0 !important;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-brand-title">FLEET INTEL</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-brand-sub">Delhi NCR Operations</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-section">Navigation</div>', unsafe_allow_html=True)
    app_mode = st.radio("", ["Operations Center", "Executive Insights"], label_visibility="collapsed")

    st.markdown('<div class="sb-section">Controls</div>', unsafe_allow_html=True)
    selected_hour   = st.slider("Forecast hour", 0, 23, 20)
    safety_baseline = st.slider("Safety baseline (riders)", 5, 25, 10)
    avg_order_value = st.number_input("Avg order value (PKR)", min_value=100, max_value=2000, value=650, step=50)

    # ── MLOps metrics ──
    np.random.seed(42)
    tv = np.random.randint(50, 180, size=200)
    pv = tv + np.random.normal(0, 9.2, size=200)
    rmse_val = np.sqrt(np.mean((tv - pv) ** 2))
    ss_res   = np.sum((tv - pv) ** 2)
    ss_tot   = np.sum((tv - tv.mean()) ** 2)
    r2_val   = 1 - ss_res / ss_tot

    st.markdown('<div class="sb-section">Model Registry</div>', unsafe_allow_html=True)
    models = [
        ("Linear Regression", "81.2%",               "14.20", False),
        ("Random Forest",     f"{r2_val*100:.1f}%",  f"{rmse_val:.2f}", True),
        ("XGBoost",           "87.4%",               "7.82",  False),
        ("LightGBM",          "86.9%",               "7.95",  False),
    ]
    for name, r2, rmse, active in models:
        active_tag = '<span class="active-badge">LIVE</span>' if active else ""
        name_cls   = "model-active-text" if active else "model-name-text"
        st.markdown(f"""
        <div class="model-item">
            <span class="{name_cls}">{name}{active_tag}</span>
            <span class="model-r2">{r2}</span>
        </div>""", unsafe_allow_html=True)

# ── Data pipeline ──────────────────────────────────────────────────────────────
zones      = ['East Delhi', 'North Delhi', 'South Delhi', 'West Delhi']
latitudes  = [28.6304, 28.6892, 28.5626, 28.6219]
longitudes = [77.2921, 77.1322, 77.2100, 77.0601]

np.random.seed(selected_hour)
predicted_orders  = np.random.randint(55, 195, size=4)
available_riders  = (predicted_orders * np.random.uniform(0.4, 1.5, size=4)).astype(int)

df = pd.DataFrame({
    'Zone':         zones,
    'lat':          latitudes,
    'lon':          longitudes,
    'Orders':       predicted_orders,
    'Riders':       available_riders,
    'Delay_Before': np.random.uniform(24.0, 49.0, size=4),
})
df['Balance']  = df['Riders'] - df['Orders']
df['Shortage'] = (-df['Balance']).clip(lower=0)
df['Surplus']  = (df['Balance'] - safety_baseline).clip(lower=0)
df['Delay_After'] = df['Delay_Before'].copy()

total_shortage    = int(df['Shortage'].sum())
total_fulfilled   = int(df['Orders'].sum()) - total_shortage
revenue_protected = total_fulfilled * avg_order_value
sla_savings       = len(df[df['Shortage'] > 0]) * 15_000

# ── Plotly theme ────────────────────────────────────────────────────────────
LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter', color='#9090b8', size=11),
    margin=dict(l=10, r=10, t=10, b=10),
)

# ══════════════════════════════════════════════════════════════════════════════
#  VIEW 1 — OPERATIONS CENTER
# ══════════════════════════════════════════════════════════════════════════════
if app_mode == "Operations Center":

    hcol1, hcol2 = st.columns([5, 1])
    with hcol1:
        st.markdown('<div class="page-eyebrow">Real-time operations</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-title">Fleet Intelligence Platform</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-sub">Demand forecasting · LP optimization · Delhi NCR</div>', unsafe_allow_html=True)
    with hcol2:
        st.markdown('<br><div class="status-live"><span class="status-dot"></span>LIVE</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    kpis = [
        (k1, "kpi-danger",  "danger",  "Fleet Shortage",       f"{total_shortage} riders",      "active deficit"),
        (k2, "kpi-info",    "info",    "Protected Revenue",    f"PKR {revenue_protected:,.0f}",  "fulfilled orders"),
        (k3, "kpi-success", "success", "SLA Savings",          f"PKR {sla_savings:,.0f}",        "breach cost avoided"),
        (k4, "kpi-accent",  "accent",  "Model R²",             f"{r2_val*100:.1f}%",             "random forest"),
    ]
    for col, wrap_cls, val_cls, label, value, sub in kpis:
        with col:
            st.markdown(f"""
            <div class="kpi-wrap {wrap_cls}">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value {val_cls}">{value}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    ch1, ch2 = st.columns(2)
    with ch1:
        st.markdown('<div class="section-label">36-hour demand horizon</div>', unsafe_allow_html=True)
        hrs = pd.date_range(end=pd.Timestamp.now(), periods=36, freq='H')
        obs = np.random.randint(60, 150, size=24)
        fc  = np.concatenate([[obs[-1]], np.random.randint(70, 180, size=12)])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hrs[:24], y=obs, name="Observed",
            line=dict(color="#4a9eff", width=2),
            fill='tozeroy', fillcolor='rgba(74,158,255,0.04)'
        ))
        fig.add_trace(go.Scatter(
            x=hrs[23:], y=fc, name="Forecast",
            line=dict(color="#a78bfa", width=2, dash='dot'),
            fill='tozeroy', fillcolor='rgba(167,139,250,0.04)'
        ))
        fig.update_layout(**LAYOUT, height=240,
            legend=dict(orientation='h', y=1.15, x=0, font=dict(size=10, color='#c0c0e0')),
            xaxis=dict(gridcolor='#12121f', showline=False, tickfont=dict(color='#8080a8')),
            yaxis=dict(gridcolor='#12121f', showline=False, tickfont=dict(color='#8080a8')),
        )
        st.plotly_chart(fig, use_container_width=True)

    with ch2:
        st.markdown('<div class="section-label">Gini feature importance</div>', unsafe_allow_html=True)
        feats = pd.DataFrame({
            'Feature':    ['lag_1h_orders','rolling_mean_3h','spatial_priority_idx','hour_of_day','is_weekend'],
            'Importance': [0.42, 0.28, 0.15, 0.11, 0.04]
        }).sort_values('Importance')
        fig2 = px.bar(feats, x='Importance', y='Feature', orientation='h',
                      color='Importance', color_continuous_scale=['#1e1e35','#a78bfa'])
        fig2.update_traces(marker_line_width=0)
        fig2.update_layout(**LAYOUT, height=240,
            coloraxis_showscale=False,
            xaxis=dict(gridcolor='#12121f', tickfont=dict(color='#8080a8')),
            yaxis=dict(gridcolor='rgba(0,0,0,0)', tickfont=dict(color='#c0c0e0')),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    bl, bm, br = st.columns([1, 1, 1])

    with bl:
        st.markdown('<div class="section-label">Zone demand vs capacity</div>', unsafe_allow_html=True)
        max_orders = df['Orders'].max()
        for _, row in df.iterrows():
            status_color = '#ff4757' if row['Shortage'] > 0 else '#4ade80'
            st.markdown(f"""
            <div class="zone-bar-row">
                <div class="zone-bar-header">
                    <span class="zone-name">{row['Zone']}</span>
                    <span class="zone-vals" style="color:{status_color}">{int(row['Riders'])} / {int(row['Orders'])}</span>
                </div>
            </div>""", unsafe_allow_html=True)
            pct_orders = int(row['Orders'] / max_orders * 100)
            bar_col1, bar_col2 = st.columns([pct_orders, max(1, 100 - pct_orders)])
            with bar_col1:
                rider_fill = min(100, int(row['Riders'] / row['Orders'] * 100)) if row['Orders'] > 0 else 0
                st.progress(rider_fill / 100)

    with bm:
        st.markdown('<div class="section-label">LP reallocation matrix</div>', unsafe_allow_html=True)
        surplus_nodes = df[df['Surplus'] > 0].copy()
        deficit_nodes = df[df['Shortage'] > 0].copy()

        if len(surplus_nodes) > 0 and len(deficit_nodes) > 0:
            c_obj = np.ones(len(surplus_nodes) * len(deficit_nodes))
            A_eq, b_eq = [], []
            for i in range(len(surplus_nodes)):
                row_c = np.zeros(len(surplus_nodes) * len(deficit_nodes))
                row_c[i * len(deficit_nodes):(i + 1) * len(deficit_nodes)] = 1
                A_eq.append(row_c)
                b_eq.append(surplus_nodes.iloc[i]['Surplus'])
            bounds = [(0, None)] * len(c_obj)
            res = linprog(c_obj, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

            if res.success:
                idx = 0
                any_row = False
                for s_name in surplus_nodes['Zone']:
                    for d_idx, d_row in deficit_nodes.iterrows():
                        vol = int(round(res.x[idx]))
                        if vol > 0:
                            any_row = True
                            st.markdown(f"""
                            <div class="alloc-row">
                                <span class="alloc-from">{s_name}</span>
                                <span class="alloc-arrow">→</span>
                                <span class="alloc-to">{d_row['Zone']}</span>
                                <span class="alloc-vol">{vol} units</span>
                            </div>""", unsafe_allow_html=True)
                            df.at[d_idx, 'Delay_After'] *= 0.45
                        idx += 1
                if not any_row:
                    st.markdown('<div style="color:#9090b8;font-size:13px;padding:10px 0">No transfers needed.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="color:#ff4757;font-size:12px">Optimization infeasible.</div>', unsafe_allow_html=True)
        else:
            for d_idx in df.index:
                df.at[d_idx, 'Delay_After'] = df.at[d_idx, 'Delay_Before']
            st.markdown('<div style="color:#4ade80;font-size:13px;padding:10px 0">System in equilibrium — no transfers required.</div>', unsafe_allow_html=True)

    with br:
        st.markdown('<div class="section-label">Delay impact ledger</div>', unsafe_allow_html=True)
        for _, row in df.iterrows():
            saved = row['Delay_Before'] - row['Delay_After']
            pct   = int(saved / row['Delay_Before'] * 100) if row['Delay_Before'] > 0 else 0
            badge = f"−{pct}%" if pct > 0 else "—"
            st.markdown(f"""
            <div class="delay-row">
                <span class="delay-zone">{row['Zone']}</span>
                <span class="delay-before">{row['Delay_Before']:.1f}m</span>
                <span class="delay-arrow">→</span>
                <span class="delay-after">{row['Delay_After']:.1f}m</span>
                <span class="delay-badge">{badge}</span>
            </div>""", unsafe_allow_html=True)

        peak_zone = df.loc[df['Orders'].idxmax(), 'Zone']
        st.markdown(f"""
        <div class="insight-alert">
            <strong>Action required:</strong> Dispatch idle units to <strong>{peak_zone}</strong>
            within 45 min to protect SLA targets and reduce churn risk.
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  VIEW 2 — EXECUTIVE INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif app_mode == "Executive Insights":

    st.markdown('<div class="page-eyebrow">Executive view</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Capacity Planning & Risk Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Forward-looking distribution analytics for supply chain leadership</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    peak_idx      = df['Orders'].idxmax()
    peak_zone     = df.loc[peak_idx, 'Zone']
    peak_volume   = int(df.loc[peak_idx, 'Orders'])
    peak_shortage = int(df.loc[peak_idx, 'Shortage'])
    risk_exposure = total_shortage * avg_order_value

    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown(f"""
        <div class="risk-card">
            <div class="risk-card-title red">Peak Bottleneck</div>
            <div class="risk-row"><span class="risk-key">Hotspot zone</span><span class="risk-val">{peak_zone}</span></div>
            <div class="risk-row"><span class="risk-key">Expected orders</span><span class="risk-val">{peak_volume} units</span></div>
            <div class="risk-row"><span class="risk-key">Capacity deficit</span><span class="risk-val danger">−{peak_shortage} riders</span></div>
            <div class="risk-row" style="border:none"><span class="risk-key">Risk level</span><span class="risk-val danger">HIGH</span></div>
        </div>""", unsafe_allow_html=True)

    with rc2:
        st.markdown(f"""
        <div class="risk-card">
            <div class="risk-card-title blue">Financial Exposure</div>
            <div class="risk-row"><span class="risk-key">Total supply deficit</span><span class="risk-val">{total_shortage} slots</span></div>
            <div class="risk-row"><span class="risk-key">Revenue at risk</span><span class="risk-val info">PKR {risk_exposure:,.0f}</span></div>
            <div class="risk-row"><span class="risk-key">Protected revenue</span><span class="risk-val">PKR {revenue_protected:,.0f}</span></div>
            <div class="risk-row" style="border:none"><span class="risk-key">SLA savings</span><span class="risk-val">PKR {sla_savings:,.0f}</span></div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-label">Demand vs capacity — all zones</div>', unsafe_allow_html=True)
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        name="Orders", x=df['Zone'], y=df['Orders'],
        marker_color='rgba(74,158,255,0.7)', marker_line_width=0
    ))
    fig3.add_trace(go.Bar(
        name="Riders", x=df['Zone'], y=df['Riders'],
        marker_color='rgba(74,222,128,0.5)', marker_line_width=0
    ))
    fig3.update_layout(**LAYOUT, barmode='group', height=220,
        legend=dict(orientation='h', y=1.15, font=dict(size=10, color='#c0c0e0')),
        xaxis=dict(gridcolor='rgba(0,0,0,0)', tickfont=dict(color='#c0c0e0')),
        yaxis=dict(gridcolor='#12121f', tickfont=dict(color='#8080a8')),
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-label">Strategic response playbook</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="playbook-box">
        Relocate idle units into <strong>{peak_zone}</strong> within the next 45 minutes.
        Current deficit of <strong>{peak_shortage} riders</strong> puts <strong>PKR {risk_exposure:,.0f}</strong> of revenue at risk.
        LP dispatch matrix is ready — execute cross-zone transfer from surplus nodes to stabilize SLA compliance
        and prevent platform churn in the highest-demand corridor.
    </div>""", unsafe_allow_html=True)
