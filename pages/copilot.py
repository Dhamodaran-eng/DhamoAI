import streamlit as st
import pandas as pd
import snowflake.connector
from openai import OpenAI
import altair as alt

# ---------------------------------------------------
# Streamlit Config
# ---------------------------------------------------
st.set_page_config(page_title="Snowflake AI Copilot", layout="wide")
st.title("💬 Snowflake AI Copilot – Mandate Intelligence")

# ---------------------------------------------------
# Load OpenAI Client (NEW SDK)
# ---------------------------------------------------
client = OpenAI(api_key=st.secrets["openai"]["api_key"])

# ---------------------------------------------------
# Snowflake Connection Function
# ---------------------------------------------------
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
    return conn

# ---------------------------------------------------
# SQL Generator (NL → SQL)
# ---------------------------------------------------
def generate_sql_from_question(question):
    prompt = f"""
You are a Snowflake SQL expert.

Table:
TFO.TFO_SCHEMA.VW_MANDATE_PROFITABILITY_FINAL

Columns:
POSYEAR, POSMON, TOTAL_RM_COST, TOTAL_FTE, MANDATEID,
TOTAL_AUM, AVG_MONTHLY_AUM, REVENUE_AMOUNT, REVENUE_PERCENT,
ANNUALIZED_REVENUE_AMOUNT, REVENUE_PERCENT_ANNUALIZED,
RM_COST_FOR_EACH_MANDATE, PROFIT_AMOUNT, DEALNAME_COUNT

Rules:
- Generate ONLY SELECT queries
- Never use DELETE, UPDATE, INSERT, DROP
- Return ONLY SQL
- Use LIMIT when appropriate

User Question:
{question}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a financial SQL assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    sql_query = response.choices[0].message.content.strip()
    return sql_query

# ---------------------------------------------------
# Business Insight Generator
# ---------------------------------------------------
def generate_insight(question, dataframe):
    """
    Generate executive insights and next-best-actions using AI.
    """
    df_preview = dataframe.head(20).to_string(index=False) if not dataframe.empty else "No query results available."
    prompt = f"""
You are an executive financial AI advisor.

User Question:
{question}

Query Result (preview):
{df_preview}

Provide:
1. Clear insights in business terms.
2. Next-best actions a Family Office executive should consider.
3. Be concise and strategic.
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a strategic financial Copilot."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

# ---------------------------------------------------
# UI Section
# ---------------------------------------------------
st.subheader("Ask anything about Mandate Profitability")

user_question = st.text_input("Example: Show top 5 mandates by AUM")

if user_question:
    try:
        # Step 1: Generate SQL (optional)
        sql_query = generate_sql_from_question(user_question)
        st.markdown("### 🧾 Generated SQL")
        st.code(sql_query, language="sql")

        # Step 2: Execute SQL only if safe
        result_df = pd.DataFrame()
        if sql_query.lower().startswith("select"):
            try:
                conn = get_snowflake_connection()
                result_df = pd.read_sql(sql_query, conn)
                conn.close()
            except Exception as e:
                st.warning(f"Could not execute SQL: {e}")

        # Step 3: Show query results if available
        if not result_df.empty:
            st.markdown("### 📊 Query Result")
            st.dataframe(result_df, use_container_width=True)

            # Optional: Quick chart of top 5 by AUM
            if 'MANDATEID' in result_df.columns and 'TOTAL_AUM' in result_df.columns:
                top5_df = result_df.sort_values('TOTAL_AUM', ascending=False).head(5)
                chart = alt.Chart(top5_df).mark_bar().encode(
                    x='MANDATEID',
                    y='TOTAL_AUM',
                    tooltip=['MANDATEID', 'TOTAL_AUM']
                ).properties(title="Top 5 Mandates by AUM")
                st.altair_chart(chart, use_container_width=True)

        # Step 4: Generate AI Insights regardless of SQL execution
        insight = generate_insight(user_question, result_df)
        st.markdown("### 🧠 Copilot Insight & Next-Best Actions")
        st.write(insight)

    except Exception as e:
        st.error(f"❌ Error: {e}")
