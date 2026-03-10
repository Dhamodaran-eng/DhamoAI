import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
from openai import OpenAI

# ---------------------------------------------------
# Streamlit Config
# ---------------------------------------------------
st.set_page_config(page_title="NCR AI Copilot", layout="wide")
st.title("💬 NCR AI Copilot – Quality Intelligence")

# ---------------------------------------------------
# Session State
# ---------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------
# OpenAI
# ---------------------------------------------------
client = OpenAI(api_key=st.secrets["openai"]["api_key"])

# ---------------------------------------------------
# Local SQL Connection
# ---------------------------------------------------
def get_sql_engine():
    server = st.secrets["local_sql"]["server"]
    database = st.secrets["local_sql"]["database"]
    username = st.secrets["local_sql"]["username"]
    password = st.secrets["local_sql"]["password"]
    driver = st.secrets["local_sql"]["driver"]

    conn_str = (
        f"mssql+pyodbc://{username}:{password}@{server}/{database}"
        f"?driver={driver.replace(' ', '+')}"
        "&TrustServerCertificate=yes"
        "&Encrypt=no"
    )

    return create_engine(conn_str, fast_executemany=True)

@st.cache_data(ttl=300)
def run_query(sql_query):
    engine = get_sql_engine()
    return pd.read_sql(sql_query, engine)

# ---------------------------------------------------
# Detect date column automatically
# ---------------------------------------------------
def detect_date_column(df):
    for c in df.columns:
        if "date" in c.lower():
            return c
    return None

# ---------------------------------------------------
# NCR Forecast (Linear Trend)
# ---------------------------------------------------
def forecast_ncr(df):
    date_col = detect_date_column(df)
    if df.empty or date_col is None:
        return None, None

    temp = df.copy()
    temp[date_col] = pd.to_datetime(temp[date_col])

    monthly = (
        temp.groupby(temp[date_col].dt.to_period("M"))
        .size()
        .reset_index(name="count")
    )

    monthly["Month"] = monthly[date_col].astype(str)
    monthly["t"] = np.arange(len(monthly))

    # train model
    model = LinearRegression()
    model.fit(monthly[["t"]], monthly["count"])

    # future prediction (next 3 months)
    future_t = np.arange(len(monthly), len(monthly) + 3)
    preds = model.predict(future_t.reshape(-1, 1))

    future_df = pd.DataFrame({
        "Month": [f"Future-{i+1}" for i in range(3)],
        "Predicted_NCR": preds.round(0)
    })

    return monthly, future_df

# ---------------------------------------------------
# AI Insight Generator
# ---------------------------------------------------
def generate_insight(question, df):
    if df.empty:
        return "⚠️ No NCR data available."

    preview = df.head(5).to_csv(index=False)

    prompt = f"""
You are a senior Quality Engineer.

User Question:
{question}

NCR Sample Data:
{preview}

Provide:

### 📌 Summary
### 🔍 Likely Root Causes
### ⚠️ Risk Level
### 🚀 Recommended Corrective Actions
### 🛠 Preventive Measures
"""

    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        temperature=0.3
    )
    return response.output_text.strip()

# ---------------------------------------------------
# Render chat history
# ---------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------
# Chat input
# ---------------------------------------------------
user_question = st.chat_input("Ask about NCR trends, quality issues, defects…")

# ---------------------------------------------------
# Main Processing
# ---------------------------------------------------
if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        try:
            # ==================================================
            # Fetch NCR data
            # ==================================================
            sql_query = "SELECT TOP 1000 * FROM NCR"

            with st.spinner("📡 Fetching NCR data from SQL Server..."):
                df = run_query(sql_query)

            st.markdown("## 📊 NCR Data Snapshot")
            st.dataframe(df, use_container_width=True)

            # ==================================================
            # Forecast
            # ==================================================
            monthly, future = forecast_ncr(df)

            if monthly is not None:
                st.markdown("## 📈 NCR Trend")

                chart = alt.Chart(monthly).mark_line(point=True).encode(
                    x=alt.X("Month:N", title="Month"),
                    y=alt.Y("count:Q", title="NCR Count"),
                    tooltip=["Month", "count"]
                )

                st.altair_chart(chart, use_container_width=True)

                st.markdown("## 🔮 Future NCR Prediction")
                st.dataframe(future, use_container_width=True)

            # ==================================================
            # AI Recommendations
            # ==================================================
            with st.spinner("🤖 Generating quality recommendations..."):
                insight = generate_insight(user_question, df)

            st.markdown(insight)

            st.session_state.messages.append(
                {"role": "assistant", "content": insight}
            )

        except Exception as e:
            st.error(f"❌ Error: {e}")