import sqlite3
import pandas as pd

# Connect to the database
conn = sqlite3.connect('traffic_data.db')

# Read data into a pandas DataFrame (looks like an Excel sheet)
df = pd.read_sql_query("SELECT * FROM vehicle_logs", conn)

print(df)
conn.close()