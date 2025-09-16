# src/app.py
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

# Ensure we can import helper modules located in the same folder (src/)
sys.path.append(str(Path(__file__).resolve().parent))

# Local imports from src/
from preproces import preprocess_pipeline, clean_categories, feature_engineer
from ai_models import train_kmeans, detect_anomalies
from analysis import monthly_spending, spending_by_category_month, generate_insights

DATA_DEFAULT = Path(__file__).resolve().parents[1] / 'data' / 'transactions.csv'

st.set_page_config(page_title="Personal Finance Dashboard", layout="wide")
st.title("💰 Personal Finance Dashboard with AI Insights")

# --------------------
# Sidebar: data upload and controls
# --------------------
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload transactions CSV", type=['csv'])
use_sample = st.sidebar.checkbox("Use sample data", value=True)

# Load data: prefer uploaded file; else use sample CSV file
if uploaded:
    # uploaded is a file-like object -> read it into a DataFrame
    df_raw = pd.read_csv(uploaded, parse_dates=['date'])
    # run the same cleaning pipeline (we use the functions directly)
    df = clean_categories(df_raw)
    df = feature_engineer(df)
elif use_sample:
    if not DATA_DEFAULT.exists():
        st.error(f"Sample data not found at: {DATA_DEFAULT}")
        st.stop()
    df = preprocess_pipeline(str(DATA_DEFAULT))
else:
    st.info("Upload a CSV or check 'Use sample data'.")
    st.stop()

# --------------------
# Filters
# --------------------
st.sidebar.header("Filters")
min_date = st.sidebar.date_input("From", df['date'].min().date())
max_date = st.sidebar.date_input("To", df['date'].max().date())
mask = (df['date'].dt.date >= min_date) & (df['date'].dt.date <= max_date)
df = df.loc[mask].reset_index(drop=True)

st.sidebar.header("AI / Model")
n_clusters = st.sidebar.slider("KMeans clusters", 2, 10, 5)
contamination = st.sidebar.slider("Anomaly contamination (approx)", 0.01, 0.1, 0.02, step=0.01)

# --------------------
# Top KPIs
# --------------------
col1, col2, col3 = st.columns(3)
monthly = monthly_spending(df)

if not monthly.empty:
    col1.metric("Total (last month)", f"{monthly.iloc[-1]:.2f}")
    if len(monthly) >= 2:
        denom = monthly.iloc[-2] if monthly.iloc[-2] != 0 else 1.0
        delta = (monthly.iloc[-1] - monthly.iloc[-2]) / denom * 100
        col2.metric("Change vs prev month", f"{delta:.1f} %")
else:
    col1.write("No data")

# <-- FIXED: added missing closing parenthesis here
col3.write("Transactions: " + str(len(df)))

# --------------------
# Monthly trend
# --------------------
st.subheader("Monthly Spending Trend")
if not monthly.empty:
    # monthly is a Series index=month, value=amount
    monthly_df = monthly.reset_index().rename(columns={0: 'amount'}) if 'amount' not in monthly.reset_index().columns else monthly.reset_index()
    # Ensure columns are named 'month' and the series column is numeric
    if 'amount' not in monthly_df.columns:
        monthly_df.columns = ['month', 'amount']
    fig = px.line(monthly_df, x='month', y='amount', markers=True, labels={'amount': 'Total spent', 'month': 'Month'})
    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("No monthly data")

# --------------------
# Category breakdown
# --------------------
st.subheader("Spending by Category")
cat_sum = df.groupby('category')['amount'].sum().reset_index().sort_values('amount', ascending=False)
fig2 = px.bar(cat_sum, x='category', y='amount', labels={'amount': 'Total spent', 'category': 'Category'})
st.plotly_chart(fig2, use_container_width=True)

# --------------------
# Run clustering & show clusters
# --------------------
st.subheader("Clustering (typical expense groups)")
if len(df) >= 2:
    kpipe, dfc = train_kmeans(df, n_clusters=n_clusters)
    cluster_counts = dfc['cluster'].value_counts().sort_index()
    st.write("Cluster counts:")
    st.dataframe(cluster_counts.rename_axis('cluster').reset_index(name='count'))

    st.write("Sample transactions per cluster:")
    samples = dfc.groupby('cluster').apply(lambda g: g.sample(min(len(g), 3))).reset_index(drop=True)
    st.dataframe(samples[['date','category','amount','cluster','description']])
else:
    st.write("Not enough data to cluster.")

# --------------------
# Anomaly detection
# --------------------
st.subheader("Anomaly Detection (IsolationForest)")
if len(df) >= 5:
    iso, scaler, df_anom = detect_anomalies(df, contamination=contamination)
    anomalies = df_anom[df_anom['is_anomaly']].sort_values('anomaly_score')
    st.write(f"Found {len(anomalies)} anomalies")
    if not anomalies.empty:
        st.dataframe(anomalies[['date','category','amount','description','anomaly_score']])
    else:
        st.write("No anomalies detected")
else:
    st.write("Not enough data to run anomaly detection (need >=5 records).")

# --------------------
# Insights
# --------------------
st.subheader("AI Insights")
insights = generate_insights(df, anomalies_df=(anomalies if 'anomalies' in locals() else None))
for i, ins in enumerate(insights):
    st.info(f"{i+1}. {ins}")

# --------------------
# Show detailed table (transactions)
# --------------------
with st.expander("All transactions (filtered)"):
    # prefer showing df with cluster column if available
    to_show = dfc if 'dfc' in locals() else df
    cols = ['date','category','amount','description','amount_bucket']
    if 'cluster' in to_show.columns:
        cols.append('cluster')
    st.dataframe(to_show[cols])

st.write("----")
st.caption("Tip: Increase contamination to detect more anomalies, and tune cluster count to better separate spending patterns.")
