# analyze_db_schema.py

import sqlite3
import pandas as pd
import json

def analyze_db_schema(db_path):
    """
    Analyze the schema of the SQLite database and print detailed information.
    This will help us understand what tables and columns are available for our GenAI application.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get list of tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    print(f"Found {len(tables)} tables in the database:")
    
    schema_info = {}
    
    for table in tables:
        table_name = table[0]
        print(f"\n--- Table: {table_name} ---")
        
        # Get column information
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        
        print(f"Columns ({len(columns)}):")
        column_info = []
        for col in columns:
            col_id, col_name, col_type, col_notnull, col_default_value, col_pk = col
            print(f"  - {col_name} ({col_type})")
            column_info.append({
                "name": col_name,
                "type": col_type,
                "primary_key": bool(col_pk),
                "nullable": not bool(col_notnull)
            })
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        row_count = cursor.fetchone()[0]
        print(f"Row count: {row_count}")
        
        # Get sample data
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 3;")
        sample_data = cursor.fetchall()
        
        # Store table info
        schema_info[table_name] = {
            "columns": column_info,
            "row_count": row_count
        }
    
    # Save schema to a JSON file for reference
    with open('db_schema.json', 'w') as f:
        json.dump(schema_info, f, indent=2)
    
    print("\nSchema information saved to db_schema.json")
    conn.close()

if __name__ == "__main__":
    analyze_db_schema("boiler_data_all_sheets.db")