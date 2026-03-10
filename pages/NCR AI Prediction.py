import streamlit as st
import pandas as pd
import sqlalchemy
import urllib
from openai import OpenAI

st.set_page_config(layout="wide")
st.title("AI Manufacturing Quality Dashboard")

# -------------------------------------------------------
# REFRESH BUTTON (clears Streamlit cache)
# -------------------------------------------------------

if st.button("Refresh Data"):
    st.cache_data.clear()
    st.experimental_rerun()

# -------------------------------------------------------
# SQL SERVER CONNECTION
# -------------------------------------------------------

server = "prod-o3.public.c72ebf9e75f0.database.windows.net,3342"
database = "LiveO3DB"
username = "FT"
password = "ODS@ODS"
driver = "ODBC Driver 17 for SQL Server"

params = urllib.parse.quote_plus(
    f"DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}"
)

engine = sqlalchemy.create_engine(
    f"mssql+pyodbc:///?odbc_connect={params}",
    pool_pre_ping=True
)

# -------------------------------------------------------
# LOAD MONTHLY NCR TREND
# -------------------------------------------------------

@st.cache_data(ttl=3600)
def load_monthly_trends():

    query = """
    SELECT 
        CAST(DATEFROMPARTS(YEAR(CreatedOn), MONTH(CreatedOn), 1) AS DATE) AS Month,
        COUNT(*) AS NCR_Count
    FROM NCR
    WHERE CreatedOn IS NOT NULL
    GROUP BY DATEFROMPARTS(YEAR(CreatedOn), MONTH(CreatedOn), 1)
    ORDER BY Month
    """

    df = pd.read_sql(query, engine)

    df["Month"] = pd.to_datetime(df["Month"])

    return df


# -------------------------------------------------------
# LOAD TOP DEFECTS
# -------------------------------------------------------

@st.cache_data(ttl=3600)
def load_top_defects():

    query = """
    SELECT TOP 10
        ISNULL(FailureMode,'Unknown') AS FailureMode,
        COUNT(*) AS Count
    FROM NCR
    GROUP BY FailureMode
    ORDER BY Count DESC
    """

    return pd.read_sql(query, engine)


# -------------------------------------------------------
# LOAD ROOT CAUSES
# -------------------------------------------------------

@st.cache_data(ttl=3600)
def load_root_causes():

    query = """
    SELECT TOP 10
        ISNULL(FailureModeDetail,'Unknown') AS FailureModeDetail,
        COUNT(*) AS Count
    FROM NCR
    GROUP BY FailureModeDetail
    ORDER BY Count DESC
    """

    return pd.read_sql(query, engine)


# -------------------------------------------------------
# LOAD RECENT NCR
# -------------------------------------------------------

@st.cache_data(ttl=3600)
def load_recent_ncr():

    query = """
    SELECT TOP 50
        NcrId,
        ProdDesc,
        FailureMode,
        NCRStatus,
        CreatedOn
    FROM NCR
    ORDER BY CreatedOn DESC
    """

    return pd.read_sql(query, engine)


# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------

monthly_df = load_monthly_trends()
defect_df = load_top_defects()
rca_df = load_root_causes()
recent_df = load_recent_ncr()

# -------------------------------------------------------
# NCR MONTHLY TREND
# -------------------------------------------------------

st.subheader("Monthly NCR Trend")

if monthly_df.empty:

    st.warning("No NCR data found")

else:

    st.dataframe(monthly_df)

    st.line_chart(monthly_df.set_index("Month"))

# -------------------------------------------------------
# TOP REPEATED DEFECTS
# -------------------------------------------------------

st.subheader("Top Repeated Defects")

if defect_df.empty:

    st.warning("No defect data available")

else:

    st.dataframe(defect_df)

# -------------------------------------------------------
# ROOT CAUSE ANALYSIS
# -------------------------------------------------------

st.subheader("Top Root Causes")

if rca_df.empty:

    st.warning("No root cause data available")

else:

    st.dataframe(rca_df)

# -------------------------------------------------------
# LATEST NCR RECORDS
# -------------------------------------------------------

st.subheader("Latest NCR Records")

if recent_df.empty:

    st.warning("No NCR records available")

else:

    st.dataframe(recent_df)

# -------------------------------------------------------
# PREDICTIVE RISK
# -------------------------------------------------------

st.subheader("Predictive Risk Assessment")

if monthly_df.empty:

    st.warning("Risk cannot be calculated")

else:

    latest = monthly_df.iloc[-1]

    risk_score = latest["NCR_Count"]

    risk_level = "Low"

    if risk_score > 50:
        risk_level = "High"

    elif risk_score > 20:
        risk_level = "Medium"

    risk_table = pd.DataFrame({
        "Metric": ["Last Month NCR", "Risk Level"],
        "Value": [risk_score, risk_level]
    })

    st.table(risk_table)

# -------------------------------------------------------
# AI INSIGHTS
# -------------------------------------------------------

st.subheader("AI Insights")

try:

    client = OpenAI(api_key=st.secrets["openai"]["api_key"])

    prompt = f"""
    Monthly NCR Trend:
    {monthly_df.tail(3).to_string()}

    Top Defects:
    {defect_df.head(5).to_string()}

    Root Causes:
    {rca_df.head(5).to_string()}

    Provide short insights and recommended actions.
    """

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    st.write(response.choices[0].message.content)

except:

    st.warning("AI insights unavailable")

# -------------------------------------------------------
# NEXT BEST ACTIONS
# -------------------------------------------------------

st.subheader("Recommended Corrective Actions")

try:

    action_prompt = f"""
    Top Defects:
    {defect_df.head(3).to_string()}

    Root Causes:
    {rca_df.head(3).to_string()}

    Provide 5 short corrective actions for quality engineers.
    """

    actions = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": action_prompt}]
    )

    st.success(actions.choices[0].message.content)

except:

    st.warning("AI recommendations unavailable")