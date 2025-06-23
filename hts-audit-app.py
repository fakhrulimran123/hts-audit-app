import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(layout="wide")
st.title("📊 HTS Branch Audit Dashboard")

# Upload Excel file
uploaded_file = st.file_uploader("Upload your audit Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Load branch coordinates
    try:
        location_df = pd.read_csv("branch_locations.csv")
        df = df.merge(location_df, how='left', on='BRANCH')
    except FileNotFoundError:
        st.error("📍 branch_locations.csv not found! Please upload it alongside the app.")

    # Data Cleaning
    df['DATE'] = pd.to_datetime(df['DATE'], dayfirst=True)
    df['DOWNLOAD SPEED (Mbps)'] = pd.to_numeric(df['DOWNLOAD SPEED (Mbps)'], errors='coerce')
    df['UPLOAD SPEED (Mbps)'] = pd.to_numeric(df['UPLOAD SPEED (Mbps)'], errors='coerce')

    st.subheader("Raw Data Preview")
    st.dataframe(df)

    st.markdown("---")

    # Map Section
    st.subheader("📍 Branch Location Map")
    map_df = df.dropna(subset=['LATITUDE', 'LONGITUDE'])
    st.map(map_df[['LATITUDE', 'LONGITUDE']])

    # Missing Audit Years Detection
    st.subheader("🕵️ Branches Missing Audit Years")
    all_branches = df['BRANCH'].unique()
    expected_years = [2022, 2023, 2024, 2025]
    df['YEAR'] = df['DATE'].dt.year
    existing_pairs = set(zip(df['BRANCH'], df['YEAR']))

    missing_records = [
        {'BRANCH': branch, 'MISSING YEAR': year}
        for branch in all_branches for year in expected_years
        if (branch, year) not in existing_pairs
    ]

    if not missing_records:
        st.success("✅ All branches have complete audit records for 2022–2025!")
    else:
        missing_df = pd.DataFrame(missing_records)
        st.warning(f"⚠️ Found {len(missing_records)} missing audit entries.")
        selected_branch = st.selectbox("🔍 Filter missing data by branch:", ["All"] + list(missing_df['BRANCH'].unique()))
        if selected_branch != "All":
            missing_df = missing_df[missing_df['BRANCH'] == selected_branch]
        st.dataframe(missing_df.reset_index(drop=True))
    st.markdown("---")

    # Download Speed Trend Over Time by Branch
    st.subheader("📈 Download Speed Trend Over Time by Branch")
    selected_branches = st.multiselect("Select Branch(es):", options=df['BRANCH'].unique(), default=df['BRANCH'].unique())
    branch_filtered_df = df[df['BRANCH'].isin(selected_branches)]

    if not branch_filtered_df.empty:
        trend_data = branch_filtered_df.groupby(['DATE', 'BRANCH'])['DOWNLOAD SPEED (Mbps)'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=trend_data, x='DATE', y='DOWNLOAD SPEED (Mbps)', hue='BRANCH', marker='o', ax=ax)
        ax.set_title("Average Download Speed Over Time by Branch")
        ax.set_ylabel("Download Speed (Mbps)")
        ax.set_xlabel("Date")
        ax.legend(title="Branch", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
        st.pyplot(fig)

    # Average Download Speed by ISP
    st.subheader("🛁 Average Download Speed by ISP")
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.barplot(x='ISP', y='DOWNLOAD SPEED (Mbps)', data=df, estimator=np.mean, ci=None, ax=ax1)
    ax1.set_title('Average Download Speed by ISP')
    st.pyplot(fig1)
    st.markdown("---")

    # Download Speed Trend Over Time by ISP
    st.subheader("📈 Download Speed Trend Over Time")
    selected_isps = st.multiselect("Select ISP(s) to display:", options=df['ISP'].unique(), default=df['ISP'].unique())
    filtered_df = df[df['ISP'].isin(selected_isps)]

    if not filtered_df.empty:
        speed_trend = filtered_df.groupby(['DATE', 'ISP'])['DOWNLOAD SPEED (Mbps)'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=speed_trend, x='DATE', y='DOWNLOAD SPEED (Mbps)', hue='ISP', marker='o', ax=ax)
        ax.set_title("Average Download Speed Over Time by ISP")
        ax.set_ylabel("Download Speed (Mbps)")
        ax.set_xlabel("Date")
        ax.legend(title="ISP")
        ax.grid(True)
        st.pyplot(fig)
    st.markdown("---")

    # Printer Model Count
    st.subheader("🖨️ Printer Model Usage")
    printer_counts = df['PRINTER BRAND'].value_counts()
    st.bar_chart(printer_counts)
    st.markdown("---")
