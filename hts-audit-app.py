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
    
    # Data Cleaning
    df['DATE'] = pd.to_datetime(df['DATE'], dayfirst=True)
    df['DOWNLOAD SPEED (Mbps)'] = pd.to_numeric(df['DOWNLOAD SPEED (Mbps)'], errors='coerce')
    df['UPLOAD SPEED (Mbps)'] = pd.to_numeric(df['UPLOAD SPEED (Mbps)'], errors='coerce')

    st.subheader("Raw Data Preview")
    st.dataframe(df)

    st.markdown("---")

    # Section: 🕵️ Missing Audit Years Detection
    st.subheader("🕵️ Branches Missing Audit Years")

    # Get all unique branches and expected years
    all_branches = df['BRANCH'].unique()
    expected_years = [2022, 2023, 2024, 2025]  # Adjust if your range is different

    # Create a record of existing (branch, year) pairs
    df['YEAR'] = df['DATE'].dt.year
    existing_pairs = set(zip(df['BRANCH'], df['YEAR']))

    # Detect missing combinations
    missing_records = []

    for branch in all_branches:
        for year in expected_years:
            if (branch, year) not in existing_pairs:
                missing_records.append({
                    'BRANCH': branch,
                    'MISSING YEAR': year
                })

    # Display results
    if not missing_records:
        st.success("✅ All branches have complete audit records for 2021–2024!")
    else:
        missing_df = pd.DataFrame(missing_records)
        st.warning(f"⚠️ Found {len(missing_records)} missing audit entries.")
        
        # Optional filter by branch
        selected_branch = st.selectbox("🔍 Filter missing data by branch:", ["All"] + list(missing_df['BRANCH'].unique()))
        if selected_branch != "All":
            missing_df = missing_df[missing_df['BRANCH'] == selected_branch]

    st.dataframe(missing_df.reset_index(drop=True))
    st.markdown("---")

    # Section: 📈 Download Speed Trend Over Time by Branch
    st.subheader("📈 Download Speed Trend Over Time by Branch")

    # Dropdown or multiselect for branch selection
    selected_branches = st.multiselect("Select Branch(es):", options=df['BRANCH'].unique(), default=df['BRANCH'].unique())

    # Filter data by selected branches
    branch_filtered_df = df[df['BRANCH'].isin(selected_branches)]

    if branch_filtered_df.empty:
        st.info("No data found for selected branches.")
    else:
        # Group by date and branch, calculate mean download speed
        trend_data = (
            branch_filtered_df
            .groupby(['DATE', 'BRANCH'])['DOWNLOAD SPEED (Mbps)']
            .mean()
            .reset_index()
        )

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=trend_data, x='DATE', y='DOWNLOAD SPEED (Mbps)', hue='BRANCH', marker='o', ax=ax)
    ax.set_title("Average Download Speed Over Time by Branch")
    ax.set_ylabel("Download Speed (Mbps)")
    ax.set_xlabel("Date")
    ax.legend(title="Branch", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)
    st.pyplot(fig)


    # Section: Average Download Speed by ISP
    st.subheader("📡 Average Download Speed by ISP")
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.barplot(x='ISP', y='DOWNLOAD SPEED (Mbps)', data=df, estimator=np.mean, ci=None, ax=ax1)
    ax1.set_title('Average Download Speed by ISP')
    st.pyplot(fig1)
    st.markdown("---")

    # Section: 📈 Download Speed Trend Over Time by ISP
    st.subheader("📈 Download Speed Trend Over Time")

    # Let user choose which ISP(s) to display
    selected_isps = st.multiselect("Select ISP(s) to display:", options=df['ISP'].unique(), default=df['ISP'].unique())

    # Filter the dataframe by selected ISPs
    filtered_df = df[df['ISP'].isin(selected_isps)]

    if filtered_df.empty:
        st.info("No data available for the selected ISP(s).")
    else:
        # Prepare data
        speed_trend = (
            filtered_df
            .groupby(['DATE', 'ISP'])['DOWNLOAD SPEED (Mbps)']
            .mean()
            .reset_index()
        )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=speed_trend, x='DATE', y='DOWNLOAD SPEED (Mbps)', hue='ISP', marker='o', ax=ax)
    ax.set_title("Average Download Speed Over Time by ISP")
    ax.set_ylabel("Download Speed (Mbps)")
    ax.set_xlabel("Date")
    ax.legend(title="ISP")
    ax.grid(True)
    st.pyplot(fig)
    st.markdown("---")

    # Section: Printer Model Count
    st.subheader("🖨️ Printer Model Usage")
    printer_counts = df['PRINTER BRAND'].value_counts()
    st.bar_chart(printer_counts)
    st.markdown("---")

    # Section: 🔄 Printer Change Detection by Branch
    st.subheader("🔄 Printer Change Detection by Branch")

    # Ensure sorted data
    df_sorted = df.sort_values(by=['BRANCH', 'DATE'])

    # Prepare year column
    df_sorted['YEAR'] = df_sorted['DATE'].dt.year

    # Group and detect changes
    printer_changes = []

    for branch, group in df_sorted.groupby('BRANCH'):
        printers_by_year = group.groupby('YEAR')[['PRINTER BRAND', 'PRINTER MODEL']].agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
        previous = None
        for year, row in printers_by_year.iterrows():
            current_printer = f"{row['PRINTER BRAND']} {row['PRINTER MODEL']}"
            if previous:
                if current_printer == previous:
                    status = "✅ Same printer as last year"
                else:
                    status = f"🔄 Changed from '{previous}'"
            else:
                status = "🆕 First recorded year"
            printer_changes.append({
                'BRANCH': branch,
                'YEAR': year,
                'PRINTER': current_printer,
                'STATUS': status
            })
            previous = current_printer

    printer_change_df = pd.DataFrame(printer_changes)

    # Add filters
    branches = printer_change_df['BRANCH'].unique()
    years = sorted(printer_change_df['YEAR'].unique())

    col1, col2 = st.columns(2)

    with col1:
        selected_branch = st.selectbox("Select Branch:", ["All"] + list(branches))

    with col2:
        selected_year = st.selectbox("Select Year:", ["All"] + [str(y) for y in years])

    # Apply filters
    filtered_df = printer_change_df.copy()

    if selected_branch != "All":
        filtered_df = filtered_df[filtered_df['BRANCH'] == selected_branch]

    if selected_year != "All":
        filtered_df = filtered_df[filtered_df['YEAR'] == int(selected_year)]

    st.dataframe(filtered_df.reset_index(drop=True))
    st.markdown("---")

    # Section: 🧓 Old PC Warning
    st.subheader("🧓 Old PC Warning (Manufactured Before 2015)")

    # Define threshold year
    threshold_year = 2015

    # Filter old PCs
    old_pcs = df[df['MAN. YEAR'] < threshold_year]

    if old_pcs.empty:
        st.success("✅ No old PCs found! All machines are 2015 or newer.")
    else:
        st.warning(f"⚠️ {len(old_pcs)} old PCs found (Manufactured before {threshold_year}):")
        st.dataframe(old_pcs[['BRANCH', 'COMPUTER MODEL', 'MAN. YEAR', 'PROCESSOR', 'RAM (GB)', 'HDD (GB)']])
        st.markdown("---")

    # Section: 🔁 PC Upgrade Detection
    st.subheader("🔁 PC Upgrade Detection (Branch-Level Over Time)")

    # Ensure date is sorted properly
    df_sorted = df.sort_values(by=['BRANCH', 'PC NO.', 'DATE'])

    # Group by Branch and PC NO to detect changes over time
    upgrades = []

    for (branch, pc_no), group in df_sorted.groupby(['BRANCH', 'PC NO.']):
        models = group['COMPUTER MODEL'].unique()
        years = group['MAN. YEAR'].unique()
        if len(models) > 1 or len(years) > 1:
            upgrades.append({
                'BRANCH': branch,
                'PC NO.': pc_no,
                'MODEL HISTORY': list(models),
                'YEAR HISTORY': list(years),
                'DATES RECORDED': list(group['DATE'].dt.strftime('%Y-%m-%d'))
            })

    if not upgrades:
        st.success("✅ No PC upgrades detected across audit periods.")
    else:
        st.warning(f"🔄 {len(upgrades)} PC upgrades detected!")
        st.dataframe(pd.DataFrame(upgrades))
        st.markdown("---")

    # Section: Optional Branch Filter
    st.subheader("🏢 Filter by Branch")
    selected_branch = st.selectbox("Select a Branch:", df['BRANCH'].unique())
    branch_data = df[df['BRANCH'] == selected_branch]
    st.dataframe(branch_data)

    # Optional: Export cleaned data
    st.download_button("📥 Download Cleaned Data", data=branch_data.to_csv(index=False), file_name="branch_audit.csv", mime='text/csv')