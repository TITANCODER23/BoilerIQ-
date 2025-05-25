import pandas as pd

file_path = 'Data Cleaning and Preprocessing/boiler_data_clean_all_sheets.xlsx'
df = pd.read_excel(file_path)

print(f"Shape: {df.shape}")
print(f"First few columns: {list(df.columns)[:5]}")
print(f"Missing values: {df.isna().sum().sum()}")
print(f"Sheet sources: {df['Sheet_Source'].unique()}")
print(f"Rows per sheet source:")
for sheet in df['Sheet_Source'].unique():
    count = df[df['Sheet_Source'] == sheet].shape[0]
    print(f"  {sheet}: {count} rows")
