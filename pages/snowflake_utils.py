import snowflake.connector

def run_query(query):
    try:
        conn = snowflake.connector.connect(
            user="YOUR_USERNAME",
            password="YOUR_PASSWORD",
            account="obeikan-o3ai",
            warehouse="COMPUTE_WH",
            database="TEST_DB",
            schema="PUBLIC"
        )

        cur = conn.cursor()
        cur.execute(query)
        data = cur.fetchall()
        cur.close()
        conn.close()
        return data
    
    except Exception as e:
        return f"Snowflake Error: {e}"
