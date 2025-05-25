import pandas as pd

file_path = 'Data Cleaning and Preprocessing/boiler_data_clean_by_sheet.xlsx'
excel_file = pd.ExcelFile(file_path)

print(f"Sheets in the file: {excel_file.sheet_names}")

for sheet in excel_file.sheet_names:
    df = pd.read_excel(file_path, sheet_name=sheet)
    print(f"\nSheet: {sheet}")
    print(f"Shape: {df.shape}")
    print(f"First few columns: {list(df.columns)[:5]}")
    print(f"Missing values: {df.isna().sum().sum()}")
