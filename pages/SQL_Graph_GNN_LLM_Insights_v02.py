import streamlit as st
import pandas as pd
import sqlalchemy
import urllib
import networkx as nx
from openai import OpenAI

st.title("Production Plan Risk Intelligence")

# -------------------------
# SQL CONNECTION
# -------------------------
server = "ODS-901553"
database = "LiveO3DB"
username = "sa"
password = "Welcome@123"
driver = "ODBC Driver 17 for SQL Server"

params = urllib.parse.quote_plus(
    f"DRIVER={driver};"
    f"SERVER={server};"
    f"DATABASE={database};"
    f"UID={username};"
    f"PWD={password}"
)

engine = sqlalchemy.create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

# -------------------------
# LOAD DATA
# -------------------------
st.info("Loading SQL tables...")

ncr_df = pd.read_sql("SELECT NcrId FROM NCR", engine)
prod_df = pd.read_sql("SELECT PP_Id, PT_Id, EG_Id FROM ProductionPlan", engine)
defect_df = pd.read_sql("SELECT DefectId FROM Defects", engine)

st.success("Tables loaded successfully")

# -------------------------
# BUILD GRAPH
# -------------------------
G = nx.Graph()

# Add Production Plan nodes
for _, row in prod_df.iterrows():
    G.add_node(f"PP_{row.PP_Id}", type="production")

# Add NCR nodes
for _, row in ncr_df.iterrows():
    G.add_node(f"NCR_{row.NcrId}", type="ncr")

# Add Defect nodes
for _, row in defect_df.iterrows():
    G.add_node(f"DEF_{row.DefectId}", type="defect")

# Add edges safely
for _, row in prod_df.iterrows():
    # Production → NCR
    if pd.notna(row.EG_Id):
        pp = f"PP_{row.PP_Id}"
        ncr = f"NCR_{int(row.EG_Id)}"
        if pp in G.nodes and ncr in G.nodes:
            G.add_edge(pp, ncr)
    # Production → Defect
    if pd.notna(row.PT_Id):
        pp = f"PP_{row.PP_Id}"
        defect = f"DEF_{int(row.PT_Id)}"
        if pp in G.nodes and defect in G.nodes:
            G.add_edge(pp, defect)

st.write("Graph Created")
st.write("Total Nodes:", G.number_of_nodes())
st.write("Total Edges:", G.number_of_edges())

# -------------------------
# ANALYZE PRODUCTION PLAN RISK
# -------------------------
results = []

for node, data in G.nodes(data=True):
    if data.get("type") == "production":
        neighbors = list(G.neighbors(node))
        ncr_count = 0
        defect_count = 0
        for n in neighbors:
            n_type = G.nodes[n].get("type")
            if n_type == "ncr":
                ncr_count += 1
            if n_type == "defect":
                defect_count += 1
        risk_score = ncr_count + defect_count
        results.append({
            "ProductionPlan": node,
            "NCR_Count": ncr_count,
            "Defect_Count": defect_count,
            "RiskScore": risk_score
        })

risk_df = pd.DataFrame(results)
risk_df = risk_df.sort_values("RiskScore", ascending=False)

st.subheader("Production Plan Risk Scores")
st.dataframe(risk_df)

# -------------------------
# LLM RECOMMENDATIONS
# -------------------------
st.subheader("AI Recommendations")
client = OpenAI(api_key=st.secrets["openai"]["api_key"])

# Set minimum risk threshold to show
threshold = 2
top_risks = risk_df[risk_df.RiskScore >= threshold].head(5)

if top_risks.empty:
    st.info("No high-risk Production Plans detected.")
else:
    for _, row in top_risks.iterrows():
        prompt = f"""
Production Plan {row['ProductionPlan']} has {row['NCR_Count']} NCR cases
and {row['Defect_Count']} defects connected.

Explain the risk and give a short actionable recommendation in 2-3 sentences.
"""
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        st.write(response.choices[0].message.content)