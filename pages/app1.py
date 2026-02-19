import streamlit as st
import pandas as pd
import snowflake.connector
from openai import OpenAI

# ------------------------
# Streamlit Page Setup
# ------------------------
st.set_page_config(page_title="TFO Mandate Profitability", layout="wide")
st.title("📊 TFO Mandate Profitability Results")

# ------------------------
# OpenAI Client (for insights)
# ------------------------
client = OpenAI(api_key="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")

# ------------------------
# Snowflake Connection Parameters
# ------------------------
conn_params = {
    "user": "Dhamo",
    "password": "Dhamo123456789",
    "account": "obeikan-o3ai",
    "warehouse": "TFO_WH",
    "role": "TFO_ANALYST",
    "database": "TFO",
    "schema": "TFO_SCHEMA"
}

# ------------------------
# Snowflake Query Function
# ------------------------
def run_query(sql: str) -> pd.DataFrame | str:
    try:
        conn = snowflake.connector.connect(
            user=conn_params["user"],
            password=conn_params["password"],
            account=conn_params["account"],
            warehouse=conn_params["warehouse"],
            role=conn_params["role"],
            database=conn_params["database"],
            schema=conn_params["schema"]
        )
        cs = conn.cursor()
        cs.execute(sql)
        rows = cs.fetchall()
        columns = [col[0] for col in cs.description]
        df = pd.DataFrame(rows, columns=columns)
        return df
    except Exception as e:
        return f"Snowflake Error: {e}"
    finally:
        try:
            cs.close()
            conn.close()
        except:
            pass

# ------------------------
# Generate AI Insights
# ------------------------
def generate_insights(df: pd.DataFrame, question: str) -> str:
    data_preview = df.head(20).to_csv(index=False)
    prompt = f"""
    I have the following TFO mandate profitability data:

    {data_preview}

    User Question: {question}

    Please provide:
    1. Key insights in simple language.
    2. 3–5 follow-up questions or analysis suggestions.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# ------------------------
# Main UI
# ------------------------
st.subheader("Fetch Top Profitable Mandates")

# Dynamic Inputs
year = st.number_input("Enter Year", min_value=2000, max_value=2100, value=2025, step=1)
top_n = st.number_input("Number of Top Mandates", min_value=1, max_value=100, value=10, step=1)
user_question = st.text_input("Type your question for insights (optional):")

if st.button("Get Results"):
    # Parameterized query to avoid hardcoding
    sql = f"""
    SELECT *
    FROM vw_mandate_profitability_final
    WHERE MANDATE_YEAR = {year}
    ORDER BY PROFIT DESC
    LIMIT {top_n}
    """

    st.write("**Fetching data from Snowflake...**")
    df = run_query(sql)

    if isinstance(df, pd.DataFrame) and not df.empty:
        st.subheader(f"Top {top_n} Profitable Mandates for {year}")
        st.dataframe(df)

        # Generate AI insights if user asked a question
        if user_question.strip():
            try:
                st.subheader("Insights")
                insights = generate_insights(df, user_question)
                st.write(insights)
            except Exception as e:
                st.error(f"Error generating insights: {e}")
    else:
        st.error(df if isinstance(df, str) else "No data found for the selected year.")
