# ==========================
# Streamlit + SQL → Graph → GNN → LLM Insights
# ==========================
import streamlit as st
import pandas as pd
import sqlalchemy
import networkx as nx
from node2vec import Node2Vec
import torch
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv
import openai
import urllib
import numpy as np
import pickle
import os

# --------------------------
# Streamlit title
# --------------------------
st.title("SQL → Graph → GNN → LLM Insights")

# --------------------------
# 1️⃣ SQL Connection
# --------------------------
server = "ODS-901553"      # Replace with your SQL server
database = "LiveO3DB"
username = "sa"
password = "Welcome@123"

params = urllib.parse.quote_plus(
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={server};"
    f"DATABASE={database};"
    f"UID={username};"
    f"PWD={password}"
)

engine = sqlalchemy.create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

# --------------------------
# 2️⃣ Load Tables
# --------------------------
st.info("Loading SQL tables...")

ncrs = pd.read_sql("SELECT NcrId, ProdId, ProcessId FROM NCR", engine)
defects = pd.read_sql("SELECT DefectId FROM Defects", engine)
production = pd.read_sql("SELECT PP_Id, EG_Id, PT_Id FROM ProductionPlan", engine)

st.success("Tables loaded!")

# --------------------------
# 3️⃣ Build Graph
# --------------------------
G = nx.Graph()

# Add nodes
for _, row in ncrs.iterrows():
    G.add_node(f"ncr_{row.NcrId}", type="ncr")

for _, row in defects.iterrows():
    G.add_node(f"defect_{row.DefectId}", type="defect")

for _, row in production.iterrows():
    G.add_node(f"ticket_{row.PP_Id}", type="ticket")

# Add edges based on your SQL relationships
for _, row in production.iterrows():
    # NCR → Ticket
    if not pd.isna(row.EG_Id):
        ncr_node = f"ncr_{int(row.EG_Id)}"
        ticket_node = f"ticket_{row.PP_Id}"
        if ncr_node in G.nodes() and ticket_node in G.nodes():
            G.add_edge(ncr_node, ticket_node, type="submitted")
    # Ticket → Defect
    if not pd.isna(row.PT_Id):
        ticket_node = f"ticket_{row.PP_Id}"
        defect_node = f"defect_{int(row.PT_Id)}"
        if ticket_node in G.nodes() and defect_node in G.nodes():
            G.add_edge(ticket_node, defect_node, type="related_to")

# NCR → Defect directly if ProdId exists
for _, row in ncrs.iterrows():
    if not pd.isna(row.ProdId):
        ncr_node = f"ncr_{row.NcrId}"
        defect_node = f"defect_{int(row.ProdId)}"
        if ncr_node in G.nodes() and defect_node in G.nodes():
            G.add_edge(ncr_node, defect_node, type="affects")

st.write(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# --------------------------
# 4️⃣ Node2Vec Embeddings with fallback for isolated nodes
# --------------------------
emb_file = "node_embeddings.pkl"
if os.path.exists(emb_file):
    with open(emb_file, "rb") as f:
        node2vec_model = pickle.load(f)
else:
    node2vec_model = Node2Vec(G, dimensions=32, walk_length=5, num_walks=10, workers=4)
    with st.spinner("Generating Node2Vec embeddings..."):
        node2vec_model = node2vec_model.fit(window=5, min_count=1, batch_words=4)
    with open(emb_file, "wb") as f:
        pickle.dump(node2vec_model, f)

# --------------------------
# 5️⃣ Prepare Graph for GNN safely
# --------------------------
# Only extract edges
data = from_networkx(G, group_node_attrs=None)  # group_node_attrs=None ensures no torch.cat() error

# Node features: Node2Vec for connected nodes, random small vector for isolated
embedding_dim = 32
features = []
for node in G.nodes():
    if node in node2vec_model.wv:
        features.append(node2vec_model.wv[node])
    else:
        features.append(np.random.normal(0, 0.01, embedding_dim).tolist())

data.x = torch.tensor(features, dtype=torch.float)

# --------------------------
# 6️⃣ Mock labels for demo (replace with real labels)
# --------------------------
num_nodes = len(G.nodes())
labels = np.random.randint(0, 2, size=num_nodes)
data.y = torch.tensor(labels, dtype=torch.long)

# --------------------------
# 7️⃣ Define GNN
# --------------------------
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

model_gnn = GNN(in_channels=embedding_dim, hidden_channels=16, out_channels=2)
optimizer = torch.optim.Adam(model_gnn.parameters(), lr=0.01)
loss_fn = torch.nn.NLLLoss()

# --------------------------
# 8️⃣ Train GNN
# --------------------------
st.info("Training GNN...")
with st.spinner("GNN training in progress..."):
    for epoch in range(50):
        model_gnn.train()
        optimizer.zero_grad()
        out = model_gnn(data.x, data.edge_index)
        loss = loss_fn(out, data.y)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            st.write(f"Epoch {epoch} - Loss: {loss.item()}")
st.success("GNN training complete!")

# --------------------------
# 9️⃣ LLM Insights
# --------------------------
predictions = out.argmax(dim=1)
high_risk_nodes = [node for i, node in enumerate(G.nodes()) if predictions[i] == 1]
context = f"High risk nodes (first 10): {high_risk_nodes[:10]}"
st.write(context)

# OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["openai"]["api_key"]

prompt = f"""
Analyze the following graph prediction results and suggest actionable steps:

{context}
"""

response = openai.chat.completions.create(
    model="gpt-5-mini",
    messages=[{"role": "user", "content": prompt}]
)

st.subheader("LLM Recommendations")
st.write(response.choices[0].message.content)