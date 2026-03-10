import streamlit as st
import pandas as pd
import sqlalchemy
import urllib
from openai import OpenAI

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(layout="wide", page_title="AI-Driven NCR Insights & Corrective Action Recommendations")
st.title("AI-Driven NCR Insights & Corrective Action Recommendations")

# ---------------------------
# Initialize all session state variables
# ---------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_input" not in st.session_state:
    st.session_state.last_input = ""
if "ai_insights" not in st.session_state:
    st.session_state.ai_insights = None
if "ai_actions" not in st.session_state:
    st.session_state.ai_actions = None

# ---------------------------
# Refresh Data Button
# ---------------------------
if st.button("Refresh Data"):
    st.cache_data.clear()
    st.session_state.chat_history = []
    st.session_state.last_input = ""
    st.session_state.ai_insights = None
    st.session_state.ai_actions = None
    st.success("Cache cleared! Reloading dashboard...")
    st.stop()

# ----------------------------------------
# SQL Server Connection  - STAGE SQL DB
# ---------------------------------------
# server = "prod-o3.public.c72ebf9e75f0.database.windows.net,3342"
# database = "LiveO3DB"
# username = "FT"
# password = "ODS@ODS"
# driver = "ODBC Driver 17 for SQL Server"

# ----------------------------------------
# SQL Server Connection  - LIVE SQL DB
# ---------------------------------------
server = "o3live-primerysql.2eb29dc0edfe.database.windows.net"
database = "OIG_LiveO3DB"
username = "Dhamo_RO"
password = "**Dhamo_RO123**"
driver = "ODBC Driver 17 for SQL Server"
port = 1433  # <- define SQL Server port

params = urllib.parse.quote_plus(
    f"DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}"
)


#---------------------------------------------------
#If you are running the code locally enable this code
#---------------------------------------------------
#Local 
# engine = sqlalchemy.create_engine(
#     f"mssql+pyodbc:///?odbc_connect={params}", pool_pre_ping=True
# )

#---------------------------------------------------
#If you are running the code Cloud enable this code
#---------------------------------------------------
#Cloud: 
engine = sqlalchemy.create_engine(
    f"mssql+pymssql://{username}:{password}@{server}:{port}/{database}"
)

# ---------------------------
# Load Data Functions
# ---------------------------
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

# ---------------------------
# Load Data
# ---------------------------
monthly_df = load_monthly_trends()
defect_df = load_top_defects()
rca_df = load_root_causes()
recent_df = load_recent_ncr()

# ---------------------------
# Dashboard Sections
# ---------------------------
st.subheader("Monthly NCR Trend")
if not monthly_df.empty:
    st.dataframe(monthly_df)
    st.line_chart(monthly_df.set_index("Month"))

st.subheader("Top Repeated Defects")
if not defect_df.empty:
    st.dataframe(defect_df)

st.subheader("Top Root Causes")
if not rca_df.empty:
    st.dataframe(rca_df)

st.subheader("Latest NCR Records")
if not recent_df.empty:
    st.dataframe(recent_df)

st.subheader("Predictive Risk Assessment")
if not monthly_df.empty:
    latest = monthly_df.iloc[-1]
    risk_score = latest["NCR_Count"]
    risk_level = "Low"
    if risk_score > 50:
        risk_level = "High"
    elif risk_score > 20:
        risk_level = "Medium"
    st.table(pd.DataFrame({
        "Metric": ["Last Month NCR", "Risk Level"],
        "Value": [risk_score, risk_level]
    }))

# ---------------------------
# Initialize OpenAI Client
# ---------------------------
client = OpenAI(api_key=st.secrets["openai"]["api_key"])

# ---------------------------
# AI Insights Section
# ---------------------------
st.subheader("AI Insights")
if st.session_state.ai_insights is None:
    placeholder_ai = st.empty()
    with placeholder_ai.container():
        st.markdown("""
            <div style="
                background-color:#FFF3E0;
                padding:12px;
                border-radius:10px;
                color:#E65100;
                font-weight:500;
                text-align:center;">
                🤖 AI is generating insights...
            </div>
        """, unsafe_allow_html=True)
    try:
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
        st.session_state.ai_insights = response.choices[0].message.content
    except Exception as e:
        st.session_state.ai_insights = f"AI insights unavailable: {e}"

st.markdown(f"""
    <div style="
        background-color:#BBDEFB;
        padding:12px;
        border-radius:12px;
        color:#0D47A1;
        font-weight:500;">
        {st.session_state.ai_insights}
    </div>
""", unsafe_allow_html=True)

# ---------------------------
# Recommended Corrective Actions
# ---------------------------
st.subheader("Recommended Corrective Actions")
if st.session_state.ai_actions is None:
    placeholder_actions = st.empty()
    with placeholder_actions.container():
        st.markdown("""
            <div style="
                background-color:#FFF3E0;
                padding:12px;
                border-radius:10px;
                color:#E65100;
                font-weight:500;
                text-align:center;">
                🛠️ AI is generating recommended actions...
            </div>
        """, unsafe_allow_html=True)
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
        st.session_state.ai_actions = actions.choices[0].message.content
    except Exception as e:
        st.session_state.ai_actions = f"AI recommendations unavailable: {e}"

st.markdown(f"""
    <div style="
        background-color:#C8E6C9;
        padding:12px;
        border-radius:12px;
        color:#1B5E20;
        font-weight:500;">
        {st.session_state.ai_actions}
    </div>
""", unsafe_allow_html=True)

# ---------------------------
# AI Copilot Chatbox
# ---------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("💬 AI Copilot Chatbox")

chat_container = st.container()

def display_chat():
    with chat_container:
        for i, chat in enumerate(st.session_state.chat_history):
            if chat["role"] == "user":
                st.markdown(f"""
                    <div style="
                        background-color:#E8F5E9;
                        padding:12px;
                        border-radius:12px;
                        margin:5px 0px 5px 50px;
                        color:#1B5E20;
                        font-weight:500;
                        font-size:14px;">
                        <strong>You:</strong> {chat['message']}
                    </div>
                """, unsafe_allow_html=True)
            else:
                bubble_color = "#BBDEFB" if i < len(st.session_state.chat_history)-1 else "#90CAF9"
                st.markdown(f"""
                    <div style="
                        background-color:{bubble_color};
                        padding:12px;
                        border-radius:12px;
                        margin:5px 50px 5px 0px;
                        color:#0D47A1;
                        font-weight:500;
                        font-size:14px;">
                        <strong>AI:</strong> {chat['message']}
                    </div>
                """, unsafe_allow_html=True)

display_chat()

user_question = st.text_input("Type your question here:")

if user_question and user_question != st.session_state.last_input:
    st.session_state.last_input = user_question
    st.session_state.chat_history.append({"role": "user", "message": user_question})

    placeholder = chat_container.empty()
    with placeholder.container():
        st.markdown("""
            <div style="
                background-color:#FFF3E0;
                padding:12px;
                border-radius:10px;
                color:#E65100;
                font-weight:500;
                text-align:center;">
                💬 AI is generating response...
            </div>
        """, unsafe_allow_html=True)

    try:
        chat_prompt = f"""
        Monthly NCR Trend (last 10 months):
        {monthly_df.tail(10).to_string()}

        Top Defects:
        {defect_df.head(10).to_string()}

        Top Root Causes:
        {rca_df.head(10).to_string()}

        User Question: {user_question}

        Provide a concise answer and recommended actions if relevant.
        """
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": chat_prompt}]
        )
        ai_response = response.choices[0].message.content
        st.session_state.chat_history.append({"role": "ai", "message": ai_response})
    except Exception as e:
        st.session_state.chat_history.append({"role": "ai", "message": f"AI copilot unavailable: {e}"})
    finally:
        placeholder.empty()

    display_chat()
    st.markdown("""<script>window.scrollTo(0, document.body.scrollHeight);</script>""", unsafe_allow_html=True)