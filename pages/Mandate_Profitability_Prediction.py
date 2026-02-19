import streamlit as st
import pandas as pd
import snowflake.connector
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(page_title="Mandate Forecast & Explanation", layout="wide")
st.title("🤖 AI-Driven Mandate Forecast and Explanation")

# ----------------------------
# Snowflake Connection Function
# ----------------------------
def get_snowflake_connection():
    conn = snowflake.connector.connect(
        user=st.secrets["snowflake"]["user"],
        password=st.secrets["snowflake"]["password"],
        account=st.secrets["snowflake"]["account"],
        warehouse=st.secrets["snowflake"]["warehouse"],
        database=st.secrets["snowflake"]["database"],
        schema=st.secrets["snowflake"]["schema"],
        role="TFO_ANALYST"
    )

    # Ensure correct session context
    cur = conn.cursor()
    cur.execute("USE ROLE TFO_ANALYST;")
    cur.execute("USE WAREHOUSE TFO_WH;")
    cur.execute("USE DATABASE TFO;")
    cur.execute("USE SCHEMA TFO_SCHEMA;")

    return conn

from datetime import datetime

# Current date
today = datetime.today()
current_year = today.year
current_month = today.month

# Last completed month
if current_month == 1:
    last_year = current_year - 1
    last_month = 12
else:
    last_year = current_year
    last_month = current_month - 1

# Start from November of the previous year if current month >= November, else two years back
if current_month >= 11:
    start_year = current_year
else:
    start_year = current_year - 1
start_month = 11

query = f"""
SELECT POSYEAR, POSMON, TOTAL_RM_COST, TOTAL_FTE, MANDATEID,
       TOTAL_AUM, AVG_MONTHLY_AUM, REVENUE_AMOUNT, REVENUE_PERCENT,
       ANNUALIZED_REVENUE_AMOUNT, REVENUE_PERCENT_ANNUALIZED,
       RM_COST_FOR_EACH_MANDATE, PROFIT_AMOUNT, DEALNAME_COUNT
FROM TFO.TFO_SCHEMA.VW_MANDATE_PROFITABILITY_FINAL
WHERE (POSYEAR > {start_year} OR (POSYEAR = {start_year} AND POSMON >= {start_month}))
  AND (POSYEAR < {last_year} OR (POSYEAR = {last_year} AND POSMON <= {last_month}))
ORDER BY POSYEAR, POSMON
"""

try:
    conn = get_snowflake_connection()
    df = pd.read_sql(query, conn)
    conn.close()
except Exception as e:
    st.error(f"❌ Snowflake Error: {e}")
    st.stop()

# ----------------------------
# Data Validation
# ----------------------------
if df.empty:
    st.warning("No data found for 2025.")
    st.stop()

mandates = df["MANDATEID"].unique()
selected_mandate = st.selectbox("Select MandateID:", mandates)

data = df[df["MANDATEID"] == selected_mandate].copy()
data["DATE"] = pd.to_datetime(
    data["POSYEAR"].astype(str) + "-" + data["POSMON"].astype(str) + "-01"
)

if len(data) < 3:
    st.warning("Not enough data to train the model.")
    st.stop()

# ----------------------------
# Feature Engineering
# ----------------------------
features = [
    "TOTAL_RM_COST", "TOTAL_FTE", "TOTAL_AUM", "AVG_MONTHLY_AUM",
    "REVENUE_AMOUNT", "REVENUE_PERCENT", "RM_COST_FOR_EACH_MANDATE",
    "PROFIT_AMOUNT", "DEALNAME_COUNT"
]
target = "REVENUE_PERCENT_ANNUALIZED"

X = data[features]
y = data[target]

# ----------------------------
# Model Training
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)

model.fit(X_train, y_train)
preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

# ----------------------------
# Forecast Next 3 Months Dynamically
# ----------------------------
today = datetime.today()
future_dates = [today + relativedelta(months=i) for i in range(1, 4)]

last_row = data.iloc[-1]
future_data = pd.DataFrame([last_row] * 3)
future_data["POSYEAR"] = [d.year for d in future_dates]
future_data["POSMON"] = [d.month for d in future_dates]
future_data["DATE"] = [datetime(d.year, d.month, 1) for d in future_dates]

# Slightly adjust features to simulate future variations
for col in features:
    future_data[col] *= np.random.uniform(0.98, 1.02, size=3)

future_pred = model.predict(future_data[features])
future_data["PREDICTED_REVENUE_PERCENT_ANNUALIZED"] = future_pred

# ----------------------------
# Combine Actual + Forecast
# ----------------------------
combined = pd.concat([data, future_data], ignore_index=True)

# ----------------------------
# Plot
# ----------------------------
fig = go.Figure()

# Actual line
fig.add_trace(go.Scatter(
    x=combined["DATE"][:len(data)],  # only actual dates
    y=combined[target][:len(data)],
    mode="lines+markers",
    name="Actual",
    line=dict(color="blue")
))

# Forecast line
fig.add_trace(go.Scatter(
    x=combined["DATE"][len(data):],  # forecast dates
    y=combined["PREDICTED_REVENUE_PERCENT_ANNUALIZED"][len(data):],
    mode="lines+markers",
    name="Forecast",
    line=dict(color="orange", dash="dot")
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
# AI Explanation Section
# ----------------------------
st.subheader("🧠 AI-Generated Explanation")

latest_actual = data[target].iloc[-1]
next_forecast = future_pred[0]

movement = (
    "increase" if next_forecast > latest_actual
    else "decrease" if next_forecast < latest_actual
    else "stable"
)

category_map = lambda v: "HIGH" if v > 1.5 else "MEDIUM" if v > 1.0 else "LOW"

st.markdown(f"""
### AI Explanation for Mandate {selected_mandate}

- **Current Category:** {category_map(latest_actual)}
- **Predicted Category:** {category_map(next_forecast)}
- **Movement:** {movement.upper()}
- **Current Revenue %:** {latest_actual:.2f}%
- **Predicted Next Month Revenue %:** {next_forecast:.2f}%

### Model Performance
- **MAE:** {mae:.4f}
- **R² Score:** {r2:.4f}

### Suggested Action
- Monitor AUM and deal pipeline stability
- Optimize RM cost efficiency
- Focus on high-margin mandates
""")