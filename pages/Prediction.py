import streamlit as st
import pandas as pd
import snowflake.connector
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import plotly.graph_objects as go
import numpy as np
import shap
import warnings

# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(page_title="Mandate Forecast & Explanation", layout="wide")
st.title("🤖 AI-Driven Mandate Forecast with Advanced Accuracy & Feature Insights")

# quieter warnings
warnings.filterwarnings("ignore")

# ----------------------------
# Snowflake Connection (from streamlit secrets)
# ----------------------------
user = st.secrets["snowflake"]["user"]
password = st.secrets["snowflake"]["password"]
account = st.secrets["snowflake"]["account"]
warehouse = st.secrets["snowflake"]["warehouse"]
database = st.secrets["snowflake"]["database"]
schema = st.secrets["snowflake"]["schema"]

query = """
SELECT POSYEAR, POSMON, TOTAL_RM_COST, TOTAL_FTE, MANDATEID,
       TOTAL_AUM, AVG_MONTHLY_AUM, REVENUE_AMOUNT, REVENUE_PERCENT,
       ANNUALIZED_REVENUE_AMOUNT, REVENUE_PERCENT_ANNUALIZED,
       RM_COST_FOR_EACH_MANDATE, PROFIT_AMOUNT, DEALNAME_COUNT
FROM vw_mandate_profitability_final
WHERE POSYEAR = 2025
ORDER BY POSMON
"""

conn = snowflake.connector.connect(
    user=user, password=password, account=account,
    warehouse=warehouse, database=database, schema=schema
)
df = pd.read_sql(query, conn)
conn.close()

if df.empty:
    st.warning("No data found for 2025.")
    st.stop()

# ----------------------------
# Mandate Selection
# ----------------------------
mandates = df["MANDATEID"].unique()
selected_mandate = st.selectbox("Select MandateID:", mandates)

data = df[df["MANDATEID"] == selected_mandate].copy()
data["DATE"] = pd.to_datetime(data["POSYEAR"].astype(str) + "-" + data["POSMON"].astype(str) + "-01")
data = data.sort_values("DATE")

# ----------------------------
# Lag Features
# ----------------------------
for lag in [1, 2]:
    data[f"REVENUE_PERCENT_ANNUALIZED_LAG{lag}"] = data["REVENUE_PERCENT_ANNUALIZED"].shift(lag)
data = data.dropna().reset_index(drop=True)

features = [
    "TOTAL_RM_COST", "TOTAL_FTE", "TOTAL_AUM", "AVG_MONTHLY_AUM",
    "REVENUE_AMOUNT", "REVENUE_PERCENT", "RM_COST_FOR_EACH_MANDATE",
    "PROFIT_AMOUNT", "DEALNAME_COUNT",
    "REVENUE_PERCENT_ANNUALIZED_LAG1", "REVENUE_PERCENT_ANNUALIZED_LAG2"
]
target = "REVENUE_PERCENT_ANNUALIZED"

# guard: ensure features exist
missing_feats = [f for f in features if f not in data.columns]
if missing_feats:
    st.error(f"Missing expected features in data: {missing_feats}")
    st.stop()

X = data[features].astype(float)
y = data[target].astype(float)

# ----------------------------
# Hyperparameter Tuning
# ----------------------------
st.subheader("🚀 Optimizing XGBoost Hyperparameters...")

param_grid = {
    "n_estimators": [200, 300, 400, 500],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.07],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "reg_alpha": [0, 0.1, 0.5],
    "reg_lambda": [1, 1.5, 2]
}

# If dataset is tiny, reduce n_iter and cv folds to avoid failure
n_rows = X.shape[0]
if n_rows < 12:
    # small dataset - reduce complexity of search
    tscv = TimeSeriesSplit(n_splits=max(2, min(2, n_rows - 1)))
    n_iter = 6
else:
    tscv = TimeSeriesSplit(n_splits=3)
    n_iter = 20

xgb_model = xgb.XGBRegressor(random_state=42, verbosity=0, objective="reg:squarederror")
search = RandomizedSearchCV(
    xgb_model, param_distributions=param_grid,
    n_iter=n_iter, scoring="r2", cv=tscv, verbose=0, random_state=42
)

with st.spinner("Tuning (this may take a moment for larger datasets)..."):
    # fit - RandomizedSearchCV will call fit internally
    search.fit(X, y)

best_model = search.best_estimator_

st.write("✅ Best Hyperparameters found:")
st.json(search.best_params_)

# ----------------------------
# Model Predictions & Metrics
# ----------------------------
preds = best_model.predict(X)
mae = mean_absolute_error(y, preds)
rmse = np.sqrt(mean_squared_error(y, preds))
r2 = r2_score(y, preds)

# safe MAPE (avoid division by zero)
y_nonzero = y.replace(0, np.nan)
if y_nonzero.isna().all():
    mape = np.nan
else:
    mape = (np.abs((y - preds) / y_nonzero)).mean() * 100

st.subheader("📊 Model Performance")
st.write(f"- **MAE:** {mae:.6f}")
st.write(f"- **RMSE:** {rmse:.6f}")
st.write(f"- **R² Score:** {r2:.6f}")
st.write(f"- **MAPE:** {'N/A' if np.isnan(mape) else f'{mape:.2f}%'}")

# ----------------------------
# Forecast Next 3 Months
# ----------------------------
# If you want different future months, change this block accordingly.
future_months = [11, 12, 1]
future_years = [2025, 2025, 2026]

last_row = data.iloc[-1].copy()
future_data = pd.DataFrame([last_row.to_dict()] * 3)
future_data["POSYEAR"] = future_years
future_data["POSMON"] = future_months
future_data["DATE"] = pd.to_datetime(future_data["POSYEAR"].astype(int).astype(str) + "-" + future_data["POSMON"].astype(int).astype(str) + "-01")

# propagate lags sensibly
future_data["REVENUE_PERCENT_ANNUALIZED_LAG1"] = last_row["REVENUE_PERCENT_ANNUALIZED"]
future_data["REVENUE_PERCENT_ANNUALIZED_LAG2"] = last_row.get("REVENUE_PERCENT_ANNUALIZED_LAG1", last_row["REVENUE_PERCENT_ANNUALIZED"])

# add slight random noise to numeric drivers to simulate small variation
rng = np.random.default_rng(42)
for col in ["TOTAL_RM_COST", "TOTAL_FTE", "TOTAL_AUM", "AVG_MONTHLY_AUM",
            "REVENUE_AMOUNT", "REVENUE_PERCENT", "RM_COST_FOR_EACH_MANDATE",
            "PROFIT_AMOUNT", "DEALNAME_COUNT"]:
    if col in future_data.columns:
        # small deterministic-like variations in case RNG is needed
        future_data[col] = future_data[col].astype(float) * (1 + rng.uniform(0.98, 1.02, size=len(future_data)))

# predict
future_pred = best_model.predict(future_data[features].astype(float))
future_data["PREDICTED_REVENUE_PERCENT_ANNUALIZED"] = future_pred

combined = pd.concat([data, future_data], ignore_index=True, sort=False)
combined["TYPE"] = ["Actual"] * len(data) + ["Forecast"] * len(future_data)

# ----------------------------
# Plot Actual vs Forecast
# ----------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=combined["DATE"], y=combined[target],
    mode="lines+markers", name="Actual"
))
fig.add_trace(go.Scatter(
    x=combined["DATE"], y=combined["PREDICTED_REVENUE_PERCENT_ANNUALIZED"],
    mode="lines+markers", name="Forecast", line=dict(dash="dot")
))
fig.update_layout(
    title=f"Revenue % Annualized Forecast - Mandate {selected_mandate}",
    xaxis_title="Month",
    yaxis_title="Revenue % Annualized",
    template="simple_white",
    height=500
)
st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# SHAP Feature Importance
# ----------------------------
st.subheader("🧩 Feature Importance (SHAP Values)")

# We'll try to use TreeExplainer (best for tree models). If that fails, fallback to KernelExplainer.
with st.spinner("Computing SHAP values (this may take a few seconds)..."):
    shap_values_arr = None
    try:
        # Preferred: TreeExplainer for XGBoost
        explainer = shap.TreeExplainer(best_model)
        # shap_values for regression is typically a 2D array of shape (n_samples, n_features)
        shap_values_arr = explainer.shap_values(X)
        # shap.TreeExplainer returns a numpy array for regression; if Explanation object, get .values
        if hasattr(shap_values_arr, "values"):
            shap_values_arr = shap_values_arr.values
    except Exception as e_tree:
        # fallback to KernelExplainer (slower)
        try:
            # small background sample to speed up KernelExplainer
            background = shap.sample(X, min(50, len(X)), random_state=42)
            kernel_expl = shap.KernelExplainer(best_model.predict, background)
            # limit nsamples for speed
            shap_values_arr = kernel_expl.shap_values(X, nsamples=100)
        except Exception as e_kernel:
            st.error("Failed to compute SHAP values with both TreeExplainer and KernelExplainer.")
            st.exception(e_kernel)
            shap_values_arr = None

if shap_values_arr is None:
    st.info("SHAP values not available for this model/data.")
else:
    # If shap_values are returned for every sample (n_rows x n_features)
    # convert to DataFrame and compute mean abs SHAP
    if isinstance(shap_values_arr, list):
        # rare case (e.g., multioutput), pick first array
        shap_values_arr = np.asarray(shap_values_arr[0])

    shap_values_df = pd.DataFrame(shap_values_arr, columns=features, index=X.index).abs()
    mean_abs_shap = shap_values_df.mean().sort_values(ascending=False)

    fig_shap = go.Figure(go.Bar(
        x=mean_abs_shap.values,
        y=mean_abs_shap.index,
        orientation='h'
    ))
    fig_shap.update_layout(
        title="Mean Absolute SHAP Values for Features",
        xaxis_title="Mean |SHAP value|",
        yaxis_title="Feature",
        template="simple_white",
        height=500
    )
    st.plotly_chart(fig_shap, use_container_width=True)

# ----------------------------
# AI Explanation (human-friendly)
# ----------------------------
st.subheader("🧠 AI-Generated Explanation")
latest_actual = float(data[target].iloc[-1])
next_forecast = float(future_pred[0])

movement = "increase" if next_forecast > latest_actual else "decrease" if next_forecast < latest_actual else "stable"
category_map = lambda v: "HIGH" if v > 1.5 else "MEDIUM" if v > 1.0 else "LOW"
current_cat = category_map(latest_actual)
predicted_cat = category_map(next_forecast)

st.markdown(f"""
### AI Explanation for Mandate {selected_mandate}

**Prediction Snapshot**
- **Current category:** {current_cat}  
- **Predicted category:** {predicted_cat}  
- **Movement:** {'⬆️ Increase' if movement == 'increase' else '⬇️ Decrease' if movement == 'decrease' else '➡️ Stable'}  
- **Current Revenue % (annualized):** {latest_actual:.2f}%  
- **Predicted next month Revenue %:** {next_forecast:.2f}%  

### Recommended Actions
- Review AUM and deal pipeline stability to maintain revenue momentum.  
- Focus on high-margin mandates to improve profitability.  
- Monitor cost per RM for efficiency optimization.  

### Model Performance
- **MAE:** {mae:.6f}  
- **RMSE:** {rmse:.6f}  
- **R² Score:** {r2:.6f}  
- **MAPE:** {'N/A' if np.isnan(mape) else f'{mape:.2f}%'}  
""")
