import streamlit as st
import pandas as pd
import pyodbc
import networkx as nx
from sqlalchemy import create_engine
import sys
import streamlit as st

st.write(sys.executable)


# -------------------------
# 1️⃣ SQL Server Connection
# -------------------------
st.header("SQL Server Connection")

server = st.text_input("Server", "ODS-901553")  # default instance
database = st.text_input("Database", "LiveO3DB")
username = st.text_input("Username", "sa")
password = st.text_input("Password", "Welcome@123", type="password")

if st.button("Connect to SQL Server"):
    try:
        conn = pyodbc.connect(
            "DRIVER={ODBC Driver 17 for SQL Server};"
            f"SERVER={server};"
            f"DATABASE={database};"
            f"UID={username};"
            f"PWD={password}",
            autocommit=True
        )
        st.success("✅ Connected to SQL Server")
        cursor = conn.cursor()
        cursor.execute("SELECT @@SERVERNAME")
        server_name = cursor.fetchone()[0]
        st.write(f"Server Name: {server_name}")

        # -------------------------
        # 2️⃣ Select tables to load
        # -------------------------
        tables = st.multiselect(
            "Select tables to load",
            ["NCR", "Defects", "ProductionPlan"],
            default=["NCR", "Defects"]
        )

        dfs = {}
        for table in tables:
            dfs[table] = pd.read_sql(f"SELECT TOP 100 * FROM dbo.{table}", conn)
            st.write(f"Preview of {table}:", dfs[table])

        # -------------------------
        # 3️⃣ Optional: Build Graph
        # -------------------------
        if st.checkbox("Build Graph from NCR & Defects"):
            if "NCR" in dfs and "Defects" in dfs:
                df_merged = pd.merge(dfs["NCR"], dfs["Defects"], on="DefectID", how="left")
                st.write("Merged table preview:", df_merged.head())

                G = nx.from_pandas_edgelist(
                    df_merged,
                    source="MachineID",
                    target="DefectID",
                    edge_attr=["Severity"],
                    create_using=nx.DiGraph()
                )
                st.write(f"Graph nodes: {len(G.nodes)}, edges: {len(G.edges)}")
            else:
                st.warning("Select both NCR and Defects tables to build graph")

        conn.close()
    except Exception as e:
        st.error(f"❌ SQL Server Connection Failed: {e}")

# -------------------------
# 4️⃣ Snowflake Upload
# -------------------------
st.header("Upload Tables to Snowflake")

sf_user = st.text_input("Snowflake User", "")
sf_password = st.text_input("Snowflake Password", "", type="password")
sf_account = st.text_input("Account (region)", "")
sf_database = st.text_input("Database", "")
sf_schema = st.text_input("Schema", "PUBLIC")
sf_warehouse = st.text_input("Warehouse", "")
sf_role = st.text_input("Role", "SYSADMIN")

if st.button("Upload Selected Tables to Snowflake"):
    try:
        conn_str = (
            f"snowflake://{sf_user}:{sf_password}@{sf_account}/{sf_database}/{sf_schema}"
            f"?warehouse={sf_warehouse}&role={sf_role}"
        )
        engine = create_engine(conn_str)
        st.success("✅ Connected to Snowflake")

        for table_name, df in dfs.items():
            df.to_sql(table_name, engine, index=False, if_exists="replace")
            st.write(f"✅ Uploaded {table_name} ({len(df)} rows)")

    except Exception as e:
        st.error(f"❌ Snowflake Upload Failed: {e}")