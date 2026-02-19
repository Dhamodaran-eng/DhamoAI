from PIL import Image
import streamlit as st
import snowflake.connector
import pandas as pd
import os


# =========================
# Page Config (optional title and icon)
# =========================
st.set_page_config(page_title="Snowflake Data Explorer", page_icon="❄️", layout="wide")

# =========================
# Hide default Streamlit page navigation
# =========================
hide_default_format = """
    <style>
        /* Hide the default sidebar navigation */
        [data-testid="stSidebarNav"] {display: none;}
        /* Optional: reduce sidebar padding */
        section[data-testid="stSidebar"] > div:first-child {
            padding-top: 1rem;
        }
    </style>
"""
st.markdown(hide_default_format, unsafe_allow_html=True)

# =========================
# Sidebar layout
# =========================
logo = Image.open("o3_log.png")
st.sidebar.image(logo, caption="", use_column_width=True)

# Add navigation items right AFTER the logo
st.sidebar.markdown("---")
st.sidebar.title("Navigation")
st.sidebar.page_link("app.py", label="App")
st.sidebar.page_link("pages/Home.py", label="Home")
st.sidebar.page_link("pages/MandateProfitability.py", label="MandateProfitability")
st.sidebar.page_link("pages/Prediction.py", label="Prediction")
st.sidebar.page_link("pages/Mandate_Profitability_Prediction.py", label="Prediction")
st.sidebar.page_link("pages/test_Prediction1.py", label="Prediction_test1")
st.sidebar.page_link("pages/copilot.py", label="copilot")
st.sidebar.page_link("pages/copilot2.py", label="copilot2")
st.sidebar.markdown("---")

# =========================
# Main layout
# =========================
st.title("❄️ Snowflake Data Explorer")

# 🔒 Hide credentials by using environment variables or secrets
user = st.secrets["snowflake"]["user"]
password = st.secrets["snowflake"]["password"]
account = st.secrets["snowflake"]["account"]
warehouse = st.secrets["snowflake"]["warehouse"]
database = st.secrets["snowflake"]["database"]
schema = st.secrets["snowflake"]["schema"]


st.subheader("Enter your SQL query below 👇")
query = st.text_area("SQL Query", "SELECT CURRENT_VERSION();")


if st.button("Run Query"):
    try:
        conn = snowflake.connector.connect(
            user=user,
            password=password,
            account=account,
            warehouse=warehouse,
            database=database,
            schema=schema
        )
        df = pd.read_sql(query, conn)
        conn.close()

        st.success("✅ Query executed successfully!")
        st.dataframe(df)

        if st.checkbox("Show as chart"):
            st.bar_chart(df)

         # Show bar chart if dataframe has at least 2 columns
        if df.shape[1] >= 2:
            st.subheader("📊 Line Chart")
            st.bar_chart(df.set_index(df.columns[1]))
        else:
            st.warning("Dataframe must have at least 2 columns for a bar chart.")

             # Show horizontal bar chart if dataframe has at least 3 columns
        if st.checkbox("Show horizontal bar chart"):
            if df.shape[1] >= 2:
                id_col = df.columns[0]       # e.g., Mandateid
                value_col = df.columns[1]    # e.g., revenue%

                fig, ax = plt.subplots(figsize=(8, max(4, len(df)/2)))
                ax.barh(df[y_col].astype(str), df[x_col], color="skyblue")
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f"{x_col} by {y_col}")
                st.pyplot(fig)
                

                st.pyplot(fig)
            else:
                st.warning("Dataframe must have at least 3 columns for a horizontal bar chart.")


                
    except Exception as e:
        st.error(f"❌ Error: {e}")

        
