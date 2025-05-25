import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine
import os

def handle_outliers(df, column):
    """Handle outliers in a column using the IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Replace outliers with the median
    median_val = df[column].median()
    df.loc[df[column] < lower_bound, column] = median_val
    df.loc[df[column] > upper_bound, column] = median_val
    
    return df

def clean_sheet(df):
    """Apply cleaning steps to a dataframe."""
    # 1: Fix column names (remove extra spaces, fix typos)
    df.columns = df.columns.str.strip()
    # Fix the typo in NOx column name if it exists
    if 'NOx mg/m3)' in df.columns:
        df.rename(columns={'NOx mg/m3)': 'NOx (mg/m3)'}, inplace=True)

    # 2: Handle missing values
    # For numerical columns, fill missing values with the median of the column
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # 3: Handle outliers using IQR method
    important_cols = [
        'Coal ConsumptionFeeder(MT)', 
        'Unit Generation', 
        'Boiler Efficiency',
        'SOx (mg/m3)', 
        'NOx (mg/m3)', 
        'CO (PPM)'
    ]

    for col in important_cols:
        if col in df.columns:
            df = handle_outliers(df, col)

    # 4: Create additional features for time-based analysis
    if 'Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Week'] = df['Date'].dt.isocalendar().week
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek
    
    return df

# Step 1: Load all sheets from Excel file
print("Loading data from all sheets in Excel file...")
excel_file = 'Boiler Data.xlsx'
sheet_names = pd.ExcelFile(excel_file).sheet_names
print(f"Found {len(sheet_names)} sheets: {sheet_names}")

all_sheets_data = []

for sheet_name in sheet_names:
    print(f"\nProcessing sheet: {sheet_name}")
    # Load the sheet
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Step 2: Clean and preprocess the sheet
    print("Cleaning and preprocessing data...")
    print(f"Missing values before handling: {df.isna().sum().sum()}")
    
    # Apply cleaning steps
    df_clean = clean_sheet(df)
    
    print(f"Missing values after handling: {df_clean.isna().sum().sum()}")
    
    # Add a column to identify the sheet source
    df_clean['Sheet_Source'] = sheet_name
    
    # Add to the list of processed sheets
    all_sheets_data.append(df_clean)

# Combine all sheets into a single DataFrame
combined_df = pd.concat(all_sheets_data, ignore_index=True)
print(f"\nCombined data has {combined_df.shape[0]} rows and {combined_df.shape[1]} columns")

# Step 3: Create aggregated views for weekly and monthly statistics
print("\nCreating aggregated views...")

# Weekly aggregation
if 'Week' in combined_df.columns:
    weekly_agg = combined_df.groupby(['Sheet_Source', 'Week']).agg({
        'Coal ConsumptionFeeder(MT)': 'sum',
        'Unit Generation': 'sum',
        'Boiler Efficiency': 'mean',
        'SOx (mg/m3)': 'mean',
        'NOx (mg/m3)': 'mean',
        'CO (PPM)': 'mean'
    }).reset_index()

# Monthly aggregation
if 'Month' in combined_df.columns:
    monthly_agg = combined_df.groupby(['Sheet_Source', 'Month']).agg({
        'Coal ConsumptionFeeder(MT)': 'sum',
        'Unit Generation': 'sum',
        'Boiler Efficiency': 'mean',
        'SOx (mg/m3)': 'mean',
        'NOx (mg/m3)': 'mean',
        'CO (PPM)': 'mean'
    }).reset_index()

# Step 4: Save the cleaned data to CSV and Excel
print("\nSaving cleaned data to CSV and Excel...")
if not os.path.exists('Data Cleaning and Preprocessing'):
    os.makedirs('Data Cleaning and Preprocessing')
    
combined_df.to_csv('Data Cleaning and Preprocessing/boiler_data_clean_all_sheets.csv', index=False)
combined_df.to_excel('Data Cleaning and Preprocessing/boiler_data_clean_all_sheets.xlsx', index=False)

# Also save individual cleaned sheets to Excel (in separate sheets of one file)
with pd.ExcelWriter('Data Cleaning and Preprocessing/boiler_data_clean_by_sheet.xlsx') as writer:
    for i, sheet_name in enumerate(sheet_names):
        sheet_data = all_sheets_data[i]
        sheet_data.to_excel(writer, sheet_name=sheet_name, index=False)

# Step 5: Save the data to SQLite database
print("\nSaving data to SQLite database...")
db_path = 'boiler_data_all_sheets.db'
engine = create_engine(f'sqlite:///{db_path}')

# Save the main dataframe
combined_df.to_sql('boiler_data', engine, if_exists='replace', index=False)

# Save individual sheets
for i, sheet_name in enumerate(sheet_names):
    sheet_data = all_sheets_data[i]
    sheet_data.to_sql(f'boiler_{sheet_name.lower().replace(" ", "_")}', engine, if_exists='replace', index=False)

# Save the aggregated views
if 'weekly_agg' in locals():
    weekly_agg.to_sql('weekly_stats', engine, if_exists='replace', index=False)
if 'monthly_agg' in locals():
    monthly_agg.to_sql('monthly_stats', engine, if_exists='replace', index=False)

# Verify the database
with sqlite3.connect(db_path) as conn:
    cursor = conn.cursor()
    
    # Check tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"Tables in the database: {[table[0] for table in tables]}")
    
    # Check row counts
    for table in [table[0] for table in tables]:
        cursor.execute(f"SELECT COUNT(*) FROM {table};")
        count = cursor.fetchone()[0]
        print(f"Table '{table}' has {count} rows")

print("\nData preprocessing and database creation completed successfully!")
