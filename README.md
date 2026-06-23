# Fleet Intelligence Platform

A logistics control-tower dashboard that predicts demand surges before they happen and recommends how to rebalance fleet to absorb them — combining a machine learning forecasting layer with a classical operations-research optimizer.

**Live app:** https://logistics-control-tower.streamlit.app
**Stack:** Python · Scikit-learn · SciPy · Streamlit

## The problem

On-demand delivery networks routinely hit a spatial-temporal mismatch: one zone gets flooded with orders while a neighboring zone has idle riders sitting unused. Most dashboards only show this *after* it happens, as a historical chart. This project tries to predict it ahead of time and recommend a fix before it affects delivery times.

## How it works

The platform is built around two engines:

**1. Predictive engine — demand forecasting**
A Random Forest Regressor forecasts order volume per city micro-zone over a 36-hour horizon, using spatial-temporal features such as recent order lag (`lag_1h_orders`) and short-term rolling averages (`rolling_mean_3h`). The model achieves an R² of 0.94 on held-out data.

**2. Prescriptive engine — fleet reallocation**
Once a zone is forecast to have a rider deficit or surplus, the platform models fleet allocation as a transportation problem and solves it with SciPy's linear programming solver, producing the lowest-cost routing matrix to move idle riders from surplus zones to deficit zones while respecting minimum coverage constraints.

**3. Business-facing output**
Rather than only reporting model accuracy, the dashboard translates results into operational metrics: estimated revenue protected by proactive reallocation, and estimated delay-time reduction in the corridors that would otherwise have been understaffed.

## Tech stack

- **Forecasting:** scikit-learn (Random Forest Regressor), pandas for feature engineering
- **Optimization:** SciPy linear programming (`scipy.optimize.linprog`)
- **Interface:** Streamlit

## Running locally

```bash
git clone https://github.com/Omer684/fleet-intelligence-platform.git
cd fleet-intelligence-platform
pip install -r requirements.txt
streamlit run app.py
```

## Possible extensions

- Swap the synthetic/historical demand data for a live or streaming data source
- Add a real road-network distance matrix instead of zone-to-zone abstraction
- Extend the optimizer to multi-period rolling reallocation rather than single-horizon
