import streamlit as st
import sqlite3
import pandas as pd
import time

# --- CONFIGURATION ---
st.set_page_config(
    page_title="AI Traffic Manager",
    page_icon="🚗",
    layout="wide"
)

# --- TITLE ---
st.title("🚦 Real-Time AI Traffic Analytics")
st.markdown("Monitor vehicle counts, speeds, and classification data from the computer vision system.")

# --- AUTO-REFRESH LOGIC ---
# This button allows the user to manually refresh, but we also use a loop below
if st.button('Refresh Data'):
    st.rerun()

# --- CONNECT TO DATABASE ---
def load_data():
    conn = sqlite3.connect('traffic_data.db')
    # Load data into a Pandas DataFrame
    df = pd.read_sql_query("SELECT * FROM vehicle_logs", conn)
    conn.close()
    return df

# Load the data
try:
    df = load_data()
except Exception as e:
    st.error("Database not found! Run 'Main.py' first to generate traffic data.")
    st.stop()

if df.empty:
    st.warning("No vehicles detected yet. Run 'Main.py' and let some cars pass!")
    st.stop()

# --- TOP KPI METRICS ---
# Create 3 columns for key metrics
col1, col2, col3 = st.columns(3)

with col1:
    total_count = len(df)
    st.metric(label="Total Vehicles Detected", value=total_count, delta=f"+{len(df[df.index > len(df)-5])} recent")

with col2:
    if 'speed' in df.columns:
        avg_speed = df['speed'].mean()
        st.metric(label="Average Traffic Speed", value=f"{avg_speed:.1f} km/h")

with col3:
    # Most common vehicle type
    most_common = df['vehicle_type'].mode()[0] if not df.empty else "N/A"
    st.metric(label="Dominant Vehicle Type", value=most_common)

st.markdown("---")

# --- CHARTS & GRAPHS ---
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📊 Vehicle Classification")
    # Count how many of each type (Car, Truck, etc.)
    type_counts = df['vehicle_type'].value_counts()
    st.bar_chart(type_counts)

with col_right:
    st.subheader("📈 Traffic Trend (Speed vs Time)")
    # Simple line chart of speed over the last 50 vehicles
    st.line_chart(df[['speed']].tail(50))

# --- RAW DATA TABLE ---
with st.expander("View Raw Traffic Log"):
    st.dataframe(df.sort_index(ascending=False)) # Show newest first

# Auto-refresh suggestion (Commented out to avoid lag, but good for demos)
# time.sleep(2)
# st.rerun()