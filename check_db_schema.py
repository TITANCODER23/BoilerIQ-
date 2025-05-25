import sqlite3

# Connect to the database
conn = sqlite3.connect('boiler_data_all_sheets.db')
cursor = conn.cursor()

# Get list of tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

print("Tables in the database:")
for table in tables:
    print(f"- {table[0]}")

print("\nSchema for each table:")
for table in tables:
    cursor.execute(f"PRAGMA table_info({table[0]})")
    columns = cursor.fetchall()
    print(f"\nTable: {table[0]}")
    print(f"Number of columns: {len(columns)}")
    print("First 5 columns:")
    for col in columns[:5]:
        print(f"  {col[1]} ({col[2]})")
    
    # Get row count
    cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
    row_count = cursor.fetchone()[0]
    print(f"Row count: {row_count}")

# Close the connection
conn.close()
