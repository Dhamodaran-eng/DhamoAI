import streamlit as st
import pandas as pd
import snowflake.connector
from openai import OpenAI
import altair as alt

# ---------------------------------------------------
# Streamlit Config
# ---------------------------------------------------
st.set_page_config(
    page_title="Snowflake AI Copilot",
    layout="wide"
)

st.title("💬 Snowflake AI Copilot – RM & Mandate Intelligence")
st.caption("Chat-based | Snowflake-safe | Executive Insights")

# ---------------------------------------------------
# Session State (Chat Memory)
# ---------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------
# OpenAI Client
# ---------------------------------------------------
client = OpenAI(api_key=st.secrets["openai"]["api_key"])

# ---------------------------------------------------
# Snowflake Connection
# ---------------------------------------------------
def get_snowflake_connection():
    return snowflake.connector.connect(
        user=st.secrets["snowflake"]["user"],
        password=st.secrets["snowflake"]["password"],
        account=st.secrets["snowflake"]["account"],
        warehouse=st.secrets["snowflake"]["warehouse"],
        database=st.secrets["snowflake"]["database"],
        schema=st.secrets["snowflake"]["schema"],
        role="TFO_ANALYST"
    )

# ---------------------------------------------------
# Helpers
# ---------------------------------------------------
def clean_sql(sql):
    sql = sql.strip()
    if sql.startswith("```"):
        sql = sql.replace("```sql", "").replace("```", "").strip()
    return sql

def is_safe_select(sql):
    forbidden = [
        "delete", "update", "insert", "drop",
        "alter", "truncate", "merge",
        "call", "grant", "revoke"
    ]
    sql_l = sql.lower()
    return sql_l.startswith("select") and not any(word in sql_l for word in forbidden)

# ---------------------------------------------------
# NL → SQL Generator
# ---------------------------------------------------
def generate_sql_from_question(question):
    prompt = f"""
You are a Snowflake SQL expert.

Table:
TFO.TFO_SCHEMA.VW_MANDATE_PROFITABILITY_FINAL

COLUMN SEMANTICS:
- RMID = Relationship Manager
- MANDATEID = Client Mandate

CRITICAL RULES:
- PROFIT_AMOUNT is derived → NEVER SUM(PROFIT_AMOUNT)
- Profit = SUM(REVENUE_AMOUNT) - SUM(RM_COST_FOR_EACH_MANDATE)
- TOTAL_AUM, REVENUE_AMOUNT, ANNUALIZED_REVENUE_AMOUNT → SUM
- AVG_MONTHLY_AUM, REVENUE_PERCENT → AVG
- DEALNAME_COUNT → SUM
- POSYEAR is numeric

LOGIC:
- If question mentions RM → GROUP BY RMID
- Else → GROUP BY MANDATEID

RULES:
- ONLY SELECT
- Never SELECT *
- No nested aggregates
- Use ORDER BY aliases
- Use LIMIT
- Return ONLY SQL

Available Columns:
POSYEAR, POSMON,
RMID, MANDATEID,
TOTAL_AUM, AVG_MONTHLY_AUM,
REVENUE_AMOUNT, REVENUE_PERCENT,
ANNUALIZED_REVENUE_AMOUNT,
RM_COST_FOR_EACH_MANDATE,
DEALNAME_COUNT

User Question:
{question}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": "You generate Snowflake-safe SQL only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.output_text.strip()

# ---------------------------------------------------
# Data Enrichment
# ---------------------------------------------------
def enrich_dataframe(df):
    if "TOTAL_AUM" in df.columns:
        total = df["TOTAL_AUM"].sum()
        if total > 0:
            df["AUM_SHARE_%"] = round((df["TOTAL_AUM"] / total) * 100, 2)
    return df.round(2)

# ---------------------------------------------------
# Executive Insight Generator
# ---------------------------------------------------
def generate_insight(question, df):
    if df.empty:
        return "⚠️ No data available."

    level = "Relationship Manager (RM)" if "RMID" in df.columns else "Mandate"
    preview = df.head(10).to_string(index=False)

    prompt = f"""
You are a senior Family Office AI Analyst.

Analysis Level:
{level}

User Question:
{question}

Data Sample:
{preview}

Produce insights using EXACT structure:

### 📌 Overall Summary
### 📊 Key Findings
### 🔍 Drivers & Diagnostics
### ⚠️ Risks & Observations
### 🚀 Recommendations & Next-Best Actions
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": "You are an executive financial analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.25
    )

    return response.output_text.strip()

# ---------------------------------------------------
# Render Chat History
# ---------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------
# Chat Input
# ---------------------------------------------------
user_question = st.chat_input(
    "Ask about RM performance, mandate profitability, AUM, cost, year…"
)

# ---------------------------------------------------
# Chat Processing
# ---------------------------------------------------
if user_question:
    # User message
    st.session_state.messages.append(
        {"role": "user", "content": user_question}
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        try:
            # 1️⃣ SQL Generation
            raw_sql = generate_sql_from_question(user_question)
            sql_query = clean_sql(raw_sql)

            st.markdown("### 🧾 Generated SQL")
            st.code(sql_query, language="sql")

            # 2️⃣ Execute SQL
            result_df = pd.DataFrame()
            if is_safe_select(sql_query):
                conn = get_snowflake_connection()
                result_df = pd.read_sql(sql_query, conn)
                conn.close()
            else:
                st.error("❌ Unsafe SQL detected")

            # 3️⃣ Enrich Data
            result_df = enrich_dataframe(result_df)

            # 4️⃣ Insights
            insight = generate_insight(user_question, result_df)
            st.markdown("## 🧠 AI Executive Insights")
            st.markdown(insight)

            # 5️⃣ Data Table
            if not result_df.empty:
                st.markdown("## 📊 Data Snapshot")
                st.dataframe(result_df, use_container_width=True)

            # 6️⃣ Visualization
            if "TOTAL_AUM" in result_df.columns:
                id_col = "RMID" if "RMID" in result_df.columns else "MANDATEID"
                top_df = result_df.sort_values("TOTAL_AUM", ascending=False).head(10)

                chart = alt.Chart(top_df).mark_bar().encode(
                    x=alt.X(f"{id_col}:N", title=id_col),
                    y=alt.Y("TOTAL_AUM:Q", title="Total AUM"),
                    tooltip=list(top_df.columns)
                )

                st.markdown("## 📈 Visualization")
                st.altair_chart(chart, use_container_width=True)

            # Save assistant summary to memory
            st.session_state.messages.append(
                {"role": "assistant", "content": insight}
            )

        except Exception as e:
            st.error(f"❌ Error: {e}")