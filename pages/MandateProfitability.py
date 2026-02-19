import streamlit as st
import pandas as pd
import snowflake.connector
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import plotly.graph_objects as go
import numpy as np

# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(page_title="Mandate Forecast & Insights", layout="wide")
st.title("🤖 AI-Powered Mandate Forecast with Business Recommendations")

# ----------------------------
# Snowflake Connection (FIXED)
# ----------------------------
def get_connection():
    conn = snowflake.connector.connect(
        user=st.secrets["snowflake"]["user"],
        password=st.secrets["snowflake"]["password"],
        account=st.secrets["snowflake"]["account"],
        warehouse=st.secrets["snowflake"]["warehouse"],
        database=st.secrets["snowflake"]["database"],
        schema=st.secrets["snowflake"]["schema"],
        role="TFO_ANALYST"
    )

    cur = conn.cursor()
    cur.execute("USE ROLE TFO_ANALYST;")
    cur.execute("USE WAREHOUSE TFO_WH;")
    cur.execute("USE DATABASE TFO;")
    cur.execute("USE SCHEMA TFO_SCHEMA;")

    return conn


# ----------------------------
# Load Data (FIXED QUERY)
# ----------------------------
@st.cache_data
def load_data():
    query = """
    SELECT POSYEAR, POSMON, TOTAL_RM_COST, TOTAL_FTE, MANDATEID,
           TOTAL_AUM, AVG_MONTHLY_AUM, REVENUE_AMOUNT, REVENUE_PERCENT,
           ANNUALIZED_REVENUE_AMOUNT, REVENUE_PERCENT_ANNUALIZED,
           RM_COST_FOR_EACH_MANDATE, PROFIT_AMOUNT, DEALNAME_COUNT
    FROM TFO.TFO_SCHEMA.VW_MANDATE_PROFITABILITY_FINAL
    WHERE POSYEAR >= 2020
    ORDER BY POSYEAR, POSMON
    """

    try:
        conn = get_connection()
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"❌ Snowflake Error: {e}")
        return pd.DataFrame()


df = load_data()

if df.empty:
    st.error("❌ No data available.")
    st.stop()

# ----------------------------
# Mandate Selection
# ----------------------------
mandates = sorted(df["MANDATEID"].unique())
selected_mandate = st.selectbox("Select MandateID:", mandates)

data = df[df["MANDATEID"] == selected_mandate].copy()
data["DATE"] = pd.to_datetime(
    data["POSYEAR"].astype(str) + "-" +
    data["POSMON"].astype(str) + "-01"
)
data = data.sort_values("DATE")

if len(data) < 6:
    st.warning("Not enough historical data to build a reliable model.")
    st.stop()

# ----------------------------
# Feature Engineering
# ----------------------------
for lag in [1, 2]:
    data[f"REV_PCT_ANN_LAG{lag}"] = data["REVENUE_PERCENT_ANNUALIZED"].shift(lag)

data = data.dropna()

features = [
    "TOTAL_RM_COST", "TOTAL_FTE", "TOTAL_AUM", "AVG_MONTHLY_AUM",
    "REVENUE_AMOUNT", "REVENUE_PERCENT", "RM_COST_FOR_EACH_MANDATE",
    "PROFIT_AMOUNT", "DEALNAME_COUNT",
    "REV_PCT_ANN_LAG1", "REV_PCT_ANN_LAG2"
]

target = "REVENUE_PERCENT_ANNUALIZED"

X = data[features]
y = data[target]

# ----------------------------
# Model Training + Tuning
# ----------------------------
tscv = TimeSeriesSplit(n_splits=3)

param_grid = {
    "n_estimators": [150, 250, 350],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.02, 0.04, 0.06],
    "subsample": [0.8, 1.0]
}

xgb_model = xgb.XGBRegressor(random_state=42)

search = RandomizedSearchCV(
    xgb_model,
    param_grid,
    n_iter=10,
    scoring="r2",
    cv=tscv,
    random_state=42,
    verbose=0
)

search.fit(X, y)
best_model = search.best_estimator_
preds = best_model.predict(X)

# ----------------------------
# Metrics
# ----------------------------
mae = mean_absolute_error(y, preds)
rmse = np.sqrt(mean_squared_error(y, preds))
r2 = r2_score(y, preds)
mape = np.mean(np.abs((y - preds) / y)) * 100

st.subheader("📊 Model Performance")
st.write(f"- **MAE:** {mae:.6f}")
st.write(f"- **RMSE:** {rmse:.6f}")
st.write(f"- **R² Score:** {r2:.6f}")
st.write(f"- **MAPE:** {mape:.2f}%")

# ----------------------------
# Forecast Next 3 Months
# ----------------------------
forecast_months = pd.date_range(
    start=data["DATE"].max() + pd.offsets.MonthBegin(1),
    periods=3,
    freq='MS'
)

future_rows = []
last_values = data.iloc[-1]

for date in forecast_months:
    row = last_values.copy()
    row["DATE"] = date
    row["POSYEAR"] = date.year
    row["POSMON"] = date.month
    row["REV_PCT_ANN_LAG1"] = last_values["REVENUE_PERCENT_ANNUALIZED"]
    row["REV_PCT_ANN_LAG2"] = last_values["REV_PCT_ANN_LAG1"]
    future_rows.append(row)

future_data = pd.DataFrame(future_rows)
future_preds = best_model.predict(future_data[features])
future_data["FORECAST"] = future_preds

# ----------------------------
# Plot
# ----------------------------
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data["DATE"],
    y=data[target],
    mode="lines+markers",
    name="Actual"
))

fig.add_trace(go.Scatter(
    x=future_data["DATE"],
    y=future_data["FORECAST"],
    mode="lines+markers",
    name="Forecast",
    line=dict(dash="dot")
))

fig.update_layout(
    title=f"Mandate {selected_mandate} - Revenue % Annualized Forecast",
    xaxis_title="Month",
    yaxis_title="Revenue % Annualized",
    height=450
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Feature Importance
# ----------------------------
feat_imp = pd.Series(
    best_model.feature_importances_,
    index=features
).sort_values()

fig_imp = go.Figure(go.Bar(
    x=feat_imp.values,
    y=feat_imp.index,
    orientation='h'
))

fig_imp.update_layout(
    title="Feature Importance in Forecast",
    xaxis_title="Importance Score",
    height=450
)

st.plotly_chart(fig_imp, use_container_width=True)

# ----------------------------
# AI Business Recommendation
# ----------------------------
st.subheader("🧠 AI-Generated Business Recommendations")

current_value = data[target].iloc[-1]
next_value = future_preds[0]

movement = next_value - current_value
status = "⬆️ Improving" if movement > 0 else \
         "⬇️ Declining" if movement < 0 else \
         "➡️ Stable"

def category(v):
    if v >= 1.5:
        return "HIGH"
    if v >= 1.0:
        return "MEDIUM"
    return "LOW"

st.markdown(f"""
### 📌 Insights for Mandate **{selected_mandate}**

| Item | Value |
|------|------|
| Current Revenue % Annualized | **{current_value:.2f}%** |
| Predicted Next Month | **{next_value:.2f}%** |
| Status | **{status}** |
| Current Category | **{category(current_value)}** |
| Predicted Category | **{category(next_value)}** |

#### ✅ Recommended Business Actions
- Strengthen high-margin revenue streams  
- Monitor AUM and client contribution trends  
- Improve RM cost efficiency  
- Focus on revenue-generating deal pipelines  
""")