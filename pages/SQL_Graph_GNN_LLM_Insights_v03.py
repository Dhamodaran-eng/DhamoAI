import streamlit as st
import pandas as pd
import sqlalchemy
import urllib
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from openai import OpenAI

# -------------------------
# Streamlit Setup
# -------------------------
st.set_page_config(layout="wide")
st.title("AI Manufacturing Quality Dashboard (Optimized GNN + LLM)")

# -------------------------
# Refresh Button
# -------------------------
if st.button("Refresh Data"):
    st.cache_data.clear()
    st.experimental_rerun()

# -------------------------
# SQL Server Connection
# -------------------------
server = "prod-o3.public.c72ebf9e75f0.database.windows.net,3342"
database = "LiveO3DB"
username = "FT"
password = "ODS@ODS"
driver = "ODBC Driver 17 for SQL Server"

params = urllib.parse.quote_plus(
    f"DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}"
)
engine = sqlalchemy.create_engine(
    f"mssql+pyodbc:///?odbc_connect={params}", pool_pre_ping=True
)

# -------------------------
# Load NCR Data
# -------------------------
@st.cache_data(ttl=3600)
def load_ncr():
    query = """
    SELECT NcrId, ProdDesc, FailureMode, FailureModeDetail, NCRStatus, CreatedOn
    FROM NCR
    WHERE CreatedOn IS NOT NULL
    """
    return pd.read_sql(query, engine)

ncr_df = load_ncr()

if ncr_df.empty:
    st.warning("No NCR data available")
    st.stop()

st.subheader("Latest NCR Records")
st.dataframe(ncr_df.head(50))

# -------------------------
# Graph Construction + GNN Prediction
# -------------------------
st.subheader("GNN Risk Predictions")

@st.cache_data(ttl=3600)
def build_graph_and_predict(df):
    # Create graph: Nodes = Products, Edges = shared FailureModeDetail
    G = nx.Graph()
    products = df['ProdDesc'].unique()
    G.add_nodes_from(products)

    # Vectorized edge creation
    edges = (
        df.groupby('FailureModeDetail')['ProdDesc']
        .apply(lambda x: [(x[i], x[j]) for i in range(len(x)) for j in range(i+1, len(x))])
        .explode()
        .dropna()
        .tolist()
    )

    for edge in edges:
        u, v = edge
        if G.has_edge(u, v):
            G[u][v]['weight'] += 1
        else:
            G.add_edge(u, v, weight=1)

    # Node features: number of defects per product
    node_features = pd.DataFrame({'prod': list(G.nodes())})
    node_features['feature'] = node_features['prod'].apply(lambda p: df[df['ProdDesc']==p].shape[0])
    x = torch.tensor(node_features['feature'].values, dtype=torch.float).unsqueeze(1)

    # Build edge_index for PyG
    if len(G.edges) > 0:
        edges_list = list(G.edges())
        edge_index = torch.tensor(edges_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)

    # Simple GCN
    class SimpleGCN(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super(SimpleGCN, self).__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)

        def forward(self, x, edge_index):
            if edge_index.numel() == 0:
                return x  # no edges
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            return x

    gnn = SimpleGCN(in_channels=1, hidden_channels=4, out_channels=1)
    gnn.eval()
    with torch.no_grad():
        predictions = gnn(data.x, data.edge_index)
        node_features['risk_score'] = predictions.numpy().flatten()

    node_features.sort_values('risk_score', ascending=False, inplace=True)
    return node_features

node_risks = build_graph_and_predict(ncr_df)
st.dataframe(node_risks.head(10))

# -------------------------
# LLM Insights (Async)
# -------------------------
st.subheader("AI Insights and Recommended Actions")

try:
    client = OpenAI(api_key=st.secrets["openai"]["api_key"])
    top_products = node_risks.head(5)

    prompt = f"""
    NCR GNN Predictions (Top 5 High-Risk Products):
    {top_products.to_string(index=False)}

    Latest NCR data (Top 5 records):
    {ncr_df.head(5).to_string(index=False)}

    Provide short insights and 5 recommended corrective actions for quality engineers.
    """

    with st.spinner("Generating AI insights..."):
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        st.success("Insights Generated!")
        st.write(response.choices[0].message.content)

except Exception as e:
    st.warning(f"AI insights unavailable: {e}")