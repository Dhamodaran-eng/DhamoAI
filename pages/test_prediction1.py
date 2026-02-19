# app.py
import os
import joblib
import warnings

import streamlit as st
import pandas as pd
import numpy as np
import snowflake.connector
import xgboost as xgb
import shap

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Mandate Forecast & Explanation (Fixed)", layout="wide")

# -------------------------
# Helpers
# -------------------------
def load_data_from_snowflake(query: str) -> pd.DataFrame:
    """Load data from Snowflake using secrets stored in Streamlit (no caching to avoid unhashable secrets)."""
    secrets = st.secrets["snowflake"]
    conn = snowflake.connector.connect(
        user=secrets["user"],
        password=secrets["password"],
        account=secrets["account"],
        warehouse=secrets["warehouse"],
        database=secrets["database"],
        schema=secrets["schema"],
    )
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag, rolling, pct-change and simple interaction features."""
    df = df.copy()
    df = df.sort_values("DATE").reset_index(drop=True)

    # Lag features for target
    for lag in (1, 2, 3):
        df[f"REVENUE_PERCENT_ANNUALIZED_LAG{lag}"] = df["REVENUE_PERCENT_ANNUALIZED"].shift(lag)

    # Rolling & pct-change features
    for col in ["TOTAL_AUM", "AVG_MONTHLY_AUM", "REVENUE_AMOUNT", "PROFIT_AMOUNT"]:
        if col in df.columns:
            df[f"{col}_ROLL3"] = df[col].rolling(window=3, min_periods=1).mean()
            df[f"{col}_PCT_CHANGE_1"] = df[col].pct_change(periods=1).fillna(0)

    # simple interaction
    if "TOTAL_RM_COST" in df.columns and "TOTAL_AUM" in df.columns:
        df["COST_TO_AUM"] = df["TOTAL_RM_COST"] / df["TOTAL_AUM"].replace(0, np.nan)
        df["COST_TO_AUM"] = df["COST_TO_AUM"].fillna(0)

    df = df.fillna(method="ffill").fillna(method="bfill")
    return df

def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_nonzero = np.where(y_true == 0, np.nan, y_true)
    if np.all(np.isnan(y_nonzero)):
        return np.nan
    return np.nanmean(np.abs((y_true - y_pred) / y_nonzero)) * 100

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_model(model, path: str):
    joblib.dump(model, path)

def load_model(path: str):
    return joblib.load(path)

# -------------------------
# App UI & Load
# -------------------------
st.title("🤖 Mandate Forecast & Explanation — Fixed")

QUERY = """
SELECT POSYEAR, POSMON, TOTAL_RM_COST, TOTAL_FTE, MANDATEID,
       TOTAL_AUM, AVG_MONTHLY_AUM, REVENUE_AMOUNT, REVENUE_PERCENT,
       ANNUALIZED_REVENUE_AMOUNT, REVENUE_PERCENT_ANNUALIZED,
       RM_COST_FOR_EACH_MANDATE, PROFIT_AMOUNT, DEALNAME_COUNT
FROM vw_mandate_profitability_final
WHERE POSYEAR = 2025
ORDER BY POSMON
"""

with st.spinner("Loading data from Snowflake..."):
    df_raw = load_data_from_snowflake(QUERY)

if df_raw.empty:
    st.warning("No data found for 2025.")
    st.stop()

# make DATE and select mandate
df_raw["DATE"] = pd.to_datetime(df_raw["POSYEAR"].astype(str) + "-" + df_raw["POSMON"].astype(str) + "-01")
mandates = df_raw["MANDATEID"].unique()
selected_mandate = st.selectbox("Select MandateID:", mandates)

data = df_raw[df_raw["MANDATEID"] == selected_mandate].copy()
data = data.sort_values("DATE").reset_index(drop=True)

# feature engineering
data = add_time_features(data)

# drop rows missing the required lag features
required_lags = ["REVENUE_PERCENT_ANNUALIZED_LAG1", "REVENUE_PERCENT_ANNUALIZED_LAG2"]
data = data.dropna(subset=[c for c in required_lags if c in data.columns]).reset_index(drop=True)

# define features and target
base_features = [
    "TOTAL_RM_COST", "TOTAL_FTE", "TOTAL_AUM", "AVG_MONTHLY_AUM",
    "REVENUE_AMOUNT", "REVENUE_PERCENT", "RM_COST_FOR_EACH_MANDATE",
    "PROFIT_AMOUNT", "DEALNAME_COUNT",
]
engineered = [c for c in data.columns if any(s in c for s in ["LAG", "ROLL3", "PCT_CHANGE", "COST_TO_AUM"])]
features = [f for f in (base_features + engineered) if f in data.columns]

target = "REVENUE_PERCENT_ANNUALIZED"

if len(data) < 6:
    st.warning("Not enough historical rows for reliable training. Need >= 6 rows.")
    st.stop()

X = data[features].astype(float)
y = data[target].astype(float)

st.subheader("Data & Features")
st.write(f"Rows for selected mandate: {len(data)}")
st.dataframe(data[["DATE", target] + features].tail(8))

# -------------------------
# Model path / caching
# -------------------------
model_dir = "/mnt/data/mandate_models"
ensure_dir(model_dir)
model_path = os.path.join(model_dir, f"xgb_model_mandate_{selected_mandate}.pkl")

use_cached = st.checkbox("Use cached model if available (faster)", value=True)
model_trained = None
if use_cached and os.path.exists(model_path):
    try:
        model_trained = load_model(model_path)
        st.success("Loaded cached model.")
    except Exception:
        st.warning("Failed to load cached model; will retrain.")
        model_trained = None

# -------------------------
# Train or load model (XGBoost only)
# -------------------------
if model_trained is None:
    st.subheader("🚀 Training XGBoost model...")
    # safe parameter grid
    param_grid = {
        "n_estimators": [200, 300, 400],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.01, 0.03, 0.05],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }

    tscv = TimeSeriesSplit(n_splits=2 if len(X) < 12 else 3)
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42, verbosity=0)

    # limit n_iter on small datasets
    n_iter = 6 if len(X) < 15 else 12

    search = RandomizedSearchCV(
        xgb_model,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring="r2",
        cv=tscv,
        random_state=42,
        n_jobs=-1,
        verbose=0,
        error_score="raise"  # let exceptions show for debugging if they occur
    )

    with st.spinner("Searching hyperparameters..."):
        try:
            search.fit(X, y)
            model_trained = search.best_estimator_
            st.write("✅ Best hyperparameters found:")
            st.json(search.best_params_)
            save_model(model_trained, model_path)
            st.success(f"Model trained and saved to {model_path}")
        except Exception as e:
            st.error("Model training failed. Falling back to a safe default XGBoost configuration.")
            st.exception(e)
            # fallback to a safe default fit
            xgb_model = xgb.XGBRegressor(
                objective="reg:squarederror",
                n_estimators=300,
                learning_rate=0.03,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                verbosity=0
            )
            xgb_model.fit(X, y)
            model_trained = xgb_model
            save_model(model_trained, model_path)
            st.success("Fallback model trained and saved.")

# -------------------------
# Predictions & Metrics
# -------------------------
st.subheader("📊 Model Performance (train)")
preds_train = model_trained.predict(X)
mae = mean_absolute_error(y, preds_train)
rmse = np.sqrt(mean_squared_error(y, preds_train))
r2 = r2_score(y, preds_train)
mape = safe_mape(y.values, preds_train)

st.write(f"- **MAE:** {mae:.6f}")
st.write(f"- **RMSE:** {rmse:.6f}")
st.write(f"- **R² Score:** {r2:.6f}")
st.write(f"- **MAPE:** {'N/A' if np.isnan(mape) else f'{mape:.2f}%'}")

# -------------------------
# Forecast Next 3 Months
# -------------------------
st.subheader("🔮 Forecast Next 3 Months")

# build future rows from last row (deterministic, no heavy randomization)
last_row = data.iloc[-1].copy()
forecast_periods = pd.date_range(start=last_row["DATE"] + pd.offsets.MonthBegin(1), periods=3, freq="MS")

future_rows = []
for dt in forecast_periods:
    fr = last_row.copy()
    fr["DATE"] = dt
    fr["POSYEAR"] = dt.year
    fr["POSMON"] = dt.month
    # propagate lags
    fr["REVENUE_PERCENT_ANNUALIZED_LAG1"] = last_row["REVENUE_PERCENT_ANNUALIZED"]
    fr["REVENUE_PERCENT_ANNUALIZED_LAG2"] = last_row.get("REVENUE_PERCENT_ANNUALIZED_LAG1", last_row["REVENUE_PERCENT_ANNUALIZED"])
    # keep other drivers same (you may improve with scenario inputs)
    future_rows.append(fr)

future_df = pd.DataFrame(future_rows).reset_index(drop=True)
future_X = future_df[features].astype(float)
future_preds = model_trained.predict(future_X)
future_df["PREDICTED_REVENUE_PERCENT_ANNUALIZED"] = future_preds

combined = pd.concat([data, future_df], ignore_index=True, sort=False)
combined["TYPE"] = ["Actual"] * len(data) + ["Forecast"] * len(future_df)

# plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=combined["DATE"], y=combined[target], mode="lines+markers", name="Actual"))
fig.add_trace(go.Scatter(x=combined["DATE"], y=combined["PREDICTED_REVENUE_PERCENT_ANNUALIZED"], mode="lines+markers", name="Forecast", line=dict(dash="dot")))
fig.update_layout(title=f"Revenue % Annualized - Mandate {selected_mandate}", xaxis_title="Date", yaxis_title="Revenue % Annualized", template="simple_white", height=520)
st.plotly_chart(fig, use_container_width=True)

# -------------------------
# SHAP Explanations (TreeExplainer)
# -------------------------
st.subheader("🧩 SHAP Feature Importance")

try:
    # TreeExplainer best for tree models
    explainer = shap.TreeExplainer(model_trained)
    shap_vals = explainer.shap_values(X)  # returns array (n_samples x n_features)
    # make chart of mean absolute SHAP
    shap_abs_mean = pd.DataFrame(np.abs(shap_vals), columns=features).mean().sort_values(ascending=False)
    fig_shap = go.Figure(go.Bar(x=shap_abs_mean.values, y=shap_abs_mean.index, orientation="h"))
    fig_shap.update_layout(title="Mean |SHAP value|", template="simple_white", height=520)
    st.plotly_chart(fig_shap, use_container_width=True)
except Exception as e_shap:
    st.warning("SHAP calculation failed; falling back to model.feature_importances_ if available.")
    st.write("SHAP error:", e_shap)
    try:
        fi = model_trained.feature_importances_
        fi_series = pd.Series(fi, index=features).abs().sort_values(ascending=False)
        fig_fi = go.Figure(go.Bar(x=fi_series.values, y=fi_series.index, orientation="h"))
        fig_fi.update_layout(title="Feature importances (model)", template="simple_white", height=520)
        st.plotly_chart(fig_fi, use_container_width=True)
    except Exception:
        st.info("No fallback feature importances available.")

# -------------------------
# AI Insight
# -------------------------
st.subheader("🧠 AI-Generated Insight & Recommendations")
current_val = float(data[target].iloc[-1])
next_forecast = float(future_preds[0])
movement = "Increase" if next_forecast > current_val else "Decrease" if next_forecast < current_val else "Stable"
category = lambda v: "HIGH" if v > 1.5 else "MEDIUM" if v > 1.0 else "LOW"

st.markdown(f"""
### Mandate {selected_mandate} — Snapshot
- **Current Revenue % (annualized):** {current_val:.4f}  
- **Predicted next month Revenue %:** {next_forecast:.4f}  
- **Movement:** **{movement}**
- **Current category:** **{category(current_val)}**
- **Predicted category:** **{category(next_forecast)}**

### Recommended Actions
1. Review the top drivers (SHAP / feature importance) for this mandate.  
2. If AUM volatility contributes strongly, stabilize pipeline and high-quality inflows.  
3. If RM cost drives negative performance, review RM allocation & incentives.

**Model Performance (train):**
- MAE: {mae:.6f}  
- RMSE: {rmse:.6f}  
- R²: {r2:.4f}  
- MAPE: {"N/A" if np.isnan(mape) else f"{mape:.2f}%"}  
""")
