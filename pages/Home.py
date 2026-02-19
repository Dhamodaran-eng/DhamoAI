from PIL import Image   # 👈 Add this line!
import streamlit as st
import pandas as pd
import snowflake.connector
import plotly.graph_objects as go

# =========================
# Page Config
# =========================
st.set_page_config(page_title="Home - Snowflake Explorer", page_icon="🏠", layout="wide")

# =========================
# Hide Streamlit's default page navigation
# =========================
hide_default_nav = """
    <style>
        [data-testid="stSidebarNav"] {display: none;}
        section[data-testid="stSidebar"] > div:first-child {
            padding-top: 1rem;
        }
    </style>
"""
st.markdown(hide_default_nav, unsafe_allow_html=True)

# =========================
# Sidebar Layout
# =========================
logo = Image.open("o3_log.png")
st.sidebar.image(logo, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.title("Navigation")
st.sidebar.page_link("app.py", label="App")
st.sidebar.page_link("pages/Home.py", label="Home")
st.sidebar.page_link("pages/MandateProfitability.py", label="MandateProfitability")
st.sidebar.markdown("---")

# =========================
# Title
# =========================
st.title("❄️ Snowflake Data Explorer - Home")

# =========================
# Snowflake Connection Utility
# =========================
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

    # Set session context properly
    ctx_commands = [
        "USE ROLE TFO_ANALYST;",
        "USE WAREHOUSE TFO_WH;",
        "USE DATABASE TFO;",
        "USE SCHEMA TFO_SCHEMA;"
    ]
    cur = conn.cursor()
    for cmd in ctx_commands:
        cur.execute(cmd)

    return conn

# =========================
# Query Input
# =========================
st.subheader("Enter your SQL query below 👇")
query = st.text_area(
    "SQL Query",
    "SELECT TOP 5 mandateid, revenue_percent, revenue_percent_annualized "
    "FROM TFO.TFO_SCHEMA.VW_MANDATE_PROFITABILITY_FINAL "
    "WHERE posyear = 2025 AND posmon = 9 "
    "ORDER BY revenue_percent DESC;"
)

# =========================
# Execute Query Button
# =========================
if st.button("Run Query"):
    try:
        conn = get_snowflake_connection()
        df = pd.read_sql(query, conn)
        conn.close()

        st.session_state["df"] = df
        st.success("✅ Query executed successfully!")
        st.dataframe(df, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error: {e}")

# =========================
# Use Data For Plot if Exists
# =========================
df = st.session_state.get("df", None)

if df is not None and not df.empty and df.shape[1] >= 3:
    st.subheader("📊 Revenue Percent Annualized (Top 10 Mandates)")

    x_col = df.columns[0]
    y_col = df.columns[1]
    z_col = df.columns[2]

    df[x_col] = df[x_col].astype(str)
    df = df.sort_values(by=y_col, ascending=False).head(10)

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df[x_col], y=df[y_col], mode='lines+markers',
        name=y_col, line=dict(color='blue', width=3),
        marker=dict(size=8, color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=df[x_col], y=df[z_col], mode='lines+markers',
        name=z_col, line=dict(color='green', width=3),
        marker=dict(size=8, color='green')
    ))

    fig.update_layout(
        title="Top 10 Mandates - Revenue Percent Annualized",
        xaxis_title="Mandate ID",
        yaxis_title="Revenue Percent",
        template="simple_white",
        hovermode="x unified",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================
# Bottom Dashboard Section
# =========================
tab_dashboard = st.tabs(["Dashboard"])[0]

with tab_dashboard:
    st.subheader("Top 5 Mandates (2025-09)")

    if "df" not in st.session_state:
        try:
            conn = get_snowflake_connection()
            query = """
                SELECT TOP 5 mandateid, revenue_percent, revenue_percent_annualized
                FROM TFO.TFO_SCHEMA.VW_MANDATE_PROFITABILITY_FINAL
                WHERE posyear = 2025 AND posmon = 9
                ORDER BY revenue_percent DESC;
            """
            df = pd.read_sql(query, conn)
            conn.close()

            st.session_state["df"] = df
            st.success("🚀 Data Loaded!")

        except Exception as e:
            st.error(f"❌ Error fetching data: {e}")

    df = st.session_state.get("df", None)

    if df is not None and not df.empty and df.shape[1] >= 3:
        x_col = df.columns[0]
        y_col = df.columns[1]
        z_col = df.columns[2]

        df[x_col] = df[x_col].astype(str)
        df = df.sort_values(by=y_col, ascending=False)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df[x_col], y=df[y_col], mode='lines+markers',
            name=y_col, line=dict(color='blue', width=3), marker=dict(size=8)
        ))
        fig.add_trace(go.Scatter(
            x=df[x_col], y=df[z_col], mode='lines+markers',
            name=z_col, line=dict(color='red', width=3), marker=dict(size=8)
        ))

        fig.update_layout(
            title="Top 5 Mandates - Revenue Percent Annualized",
            xaxis_title="Mandate ID",
            yaxis_title="Revenue Percent",
            template="simple_white",
            hovermode="x unified",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown("### 📋 Data Table")
        st.dataframe(df)
