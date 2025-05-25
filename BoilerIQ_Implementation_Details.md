# BoilerIQ Implementation Details

This document provides in-depth technical details about the implementation of the BoilerIQ system, complementing the main technical documentation.

## Table of Contents
1. [Code Walkthrough](#1-code-walkthrough)
2. [Data Pipeline Implementation](#2-data-pipeline-implementation)
3. [Database Schema Details](#3-database-schema-details)
4. [LangChain Agent Implementation](#4-langchain-agent-implementation)
5. [Streamlit UI Implementation](#5-streamlit-ui-implementation)
6. [Query Processing Flow](#6-query-processing-flow)
7. [Error Handling Implementation](#7-error-handling-implementation)
8. [Performance Optimization Details](#8-performance-optimization-details)

---

## 1. Code Walkthrough

### 1.1 Data Preprocessing (`data_preprocessing_all_sheets.py`)

The data preprocessing script is responsible for loading, cleaning, and transforming the raw boiler data into a structured format suitable for analysis and querying.

#### 1.1.1 Outlier Detection and Handling

```python
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
```

This function implements the Interquartile Range (IQR) method for outlier detection:
1. Calculate the first quartile (Q1) and third quartile (Q3)
2. Calculate the IQR as Q3 - Q1
3. Define the lower bound as Q1 - 1.5 * IQR
4. Define the upper bound as Q3 + 1.5 * IQR
5. Replace values outside these bounds with the median value

#### 1.1.2 Sheet Cleaning Function

```python
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
```

This function applies a series of cleaning steps to each sheet:
1. Standardize column names by removing extra spaces and fixing typos
2. Handle missing values in numerical columns by filling them with the median value
3. Handle outliers in important columns using the IQR method
4. Create additional time-based features from the Date column

#### 1.1.3 Main Processing Flow

```python
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
```

This section implements the main data loading and preprocessing flow:
1. Load the Excel file and get the list of sheet names
2. For each sheet:
   - Load the sheet into a pandas DataFrame
   - Apply the cleaning steps using the clean_sheet function
   - Add a Sheet_Source column to identify the source sheet
   - Add the cleaned sheet to a list
3. Combine all cleaned sheets into a single DataFrame

### 1.2 Database Schema Analysis (`analyse_db_schema.py`)

The database schema analysis script extracts and documents the structure of the SQLite database.

```python
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
```

This function:
1. Connects to the SQLite database
2. Retrieves the list of tables
3. For each table:
   - Gets column information (name, type, constraints)
   - Gets the row count
   - Gets sample data
4. Stores the schema information in a structured format
5. Saves the schema to a JSON file for reference

### 1.3 LangChain Agent (`agent_texttosql_main.ipynb`)

The LangChain agent is responsible for converting natural language questions into SQL queries and executing them against the database.

#### 1.3.1 Agent Initialization

```python
from langgraph.prebuilt import create_react_agent

system_prompt = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.
""".format(
    dialect=db.dialect,
    top_k=5,
)

agent = create_react_agent(
    llm,
    tools,
    prompt=system_prompt,
)
```

This code:
1. Defines a system prompt that guides the agent's behavior
2. Creates a React agent using the LLM, tools, and system prompt

#### 1.3.2 Query Processing

```python
question = "What was the value of Coal ConsumptionKPI on date 4-1-24? for Boiler 1? "

for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
```

This code:
1. Defines a natural language question
2. Streams the agent's reasoning process and actions
3. Prints each step of the agent's thought process

### 1.4 Streamlit UI (`app.py`)

The Streamlit UI provides a user-friendly interface for interacting with the BoilerIQ system.

```python
import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import initialize_agent, AgentType

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Streamlit UI
st.title("Text-to-SQL Agent 🔍")

# Connect to database
def connect_to_db():
    possible_db_paths = [
        '../boiler_data_all_sheets.db',
        os.path.join('..', 'boiler_data_all_sheets.db'),
        os.path.abspath('boiler_data_all_sheets.db'),
        os.path.abspath(os.path.join('..', 'boiler_data_all_sheets.db')),
    ]
    for db_path in possible_db_paths:
        if os.path.exists(db_path):
            try:
                db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
                return db
            except Exception as e:
                st.error(f"Failed to connect to {db_path}: {e}")
    st.stop()

db = connect_to_db()
tables = db.get_usable_table_names()
st.sidebar.write("📋 Tables in DB:")
for t in tables:
    st.sidebar.markdown(f"- `{t}`")

# Initialize LLM
llm = ChatOpenAI(model="gpt-4.1", temperature=0)

# Create Toolkit and Agent
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent = initialize_agent(
    tools=toolkit.get_tools(),
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# Input query
user_query = st.text_input("Enter your natural language question 👇")

if user_query:
    with st.spinner("Generating SQL and fetching results..."):
        try:
            result = agent.run(user_query)
            st.success("Query executed successfully!")
            st.write("**Agent's Response:**")
            st.write(result)
        except Exception as e:
            st.error(f"❌ Error: {e}")
```

This code:
1. Sets up the Streamlit UI with a title and input field
2. Connects to the SQLite database
3. Displays the available tables in the sidebar
4. Initializes the LLM and creates the agent
5. Processes user queries and displays the results

---

## 2. Data Pipeline Implementation

### 2.1 Data Flow Diagram

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│             │     │             │     │             │     │             │     │             │
│ Excel Data  │────▶│ Data Loading│────▶│ Cleaning &  │────▶│ Aggregation │────▶│ Database    │
│ (Raw)       │     │             │     │ Preprocessing│     │             │     │ Storage     │
│             │     │             │     │             │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

### 2.2 Data Aggregation Implementation

```python
# Weekly aggregation
weekly_agg = combined_df.groupby(['Sheet_Source', 'Week']).agg({
    'Coal ConsumptionFeeder(MT)': 'sum',
    'Unit Generation': 'sum',
    'Boiler Efficiency': 'mean',
    'SOx (mg/m3)': 'mean',
    'NOx (mg/m3)': 'mean',
    'CO (PPM)': 'mean'
}).reset_index()

# Monthly aggregation
monthly_agg = combined_df.groupby(['Sheet_Source', 'Month']).agg({
    'Coal ConsumptionFeeder(MT)': 'sum',
    'Unit Generation': 'sum',
    'Boiler Efficiency': 'mean',
    'SOx (mg/m3)': 'mean',
    'NOx (mg/m3)': 'mean',
    'CO (PPM)': 'mean'
}).reset_index()
```

This code creates two aggregated views:
1. Weekly aggregation: Groups data by Sheet_Source and Week
   - Sums Coal ConsumptionFeeder(MT) and Unit Generation
   - Averages Boiler Efficiency, SOx, NOx, and CO
2. Monthly aggregation: Groups data by Sheet_Source and Month
   - Sums Coal ConsumptionFeeder(MT) and Unit Generation
   - Averages Boiler Efficiency, SOx, NOx, and CO

### 2.3 Database Storage Implementation

```python
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
```

This code:
1. Saves the combined DataFrame to a table named 'boiler_data'
2. Saves each individual sheet to a separate table with a name based on the sheet name
3. Saves the weekly and monthly aggregated views to tables named 'weekly_stats' and 'monthly_stats'

---

## 3. Database Schema Details

### 3.1 Table Relationships

```
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│               │     │               │     │               │
│ boiler_bolr_1 │     │  boiler_bol2  │     │ boiler_bolr_3 │
│               │     │               │     │               │
└───────┬───────┘     └───────┬───────┘     └───────┬───────┘
        │                     │                     │
        │                     │                     │
        │                     ▼                     │
        │             ┌───────────────┐             │
        └────────────▶│  boiler_data  │◀────────────┘
                      │               │
                      └───────┬───────┘
                              │
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
        ┌───────────────┐           ┌───────────────┐
        │               │           │               │
        │ weekly_stats  │           │ monthly_stats │
        │               │           │               │
        └───────────────┘           └───────────────┘
```

### 3.2 Key Columns and Data Types

#### 3.2.1 Temporal Columns
- `Date`: DATETIME - Date of the measurement
- `Year`: INTEGER - Year extracted from Date
- `Month`: INTEGER - Month extracted from Date
- `Week`: BIGINT - Week number extracted from Date
- `Day`: INTEGER - Day extracted from Date
- `DayOfWeek`: INTEGER - Day of week (0=Monday, 6=Sunday)

#### 3.2.2 Operational Parameters
- `Coal ConsumptionFeeder(MT)`: FLOAT - Coal consumption in metric tons
- `Unit Generation`: FLOAT - Power generation output
- `Boiler Efficiency`: FLOAT - Overall boiler efficiency
- `Steam temp at boiler outlet`: FLOAT - Steam temperature
- `Steam pressure at boiler outlet(Bar)`: FLOAT - Steam pressure

#### 3.2.3 Emissions Data
- `SOx (mg/m3)`: FLOAT - Sulfur oxide emissions
- `NOx (mg/m3)`: FLOAT - Nitrogen oxide emissions
- `CO (PPM)`: FLOAT - Carbon monoxide emissions

#### 3.2.4 Efficiency Loss Factors
- `Loss Due to Dry Flue Gas`: FLOAT - Heat loss in flue gas
- `Loss due to H2O in fuel`: FLOAT - Heat loss due to moisture in fuel
- `Loss due to H2 in fuel`: FLOAT - Heat loss due to hydrogen in fuel
- `Total Unburnt Loss`: FLOAT - Heat loss due to unburnt fuel
- `Radiation Loss`: FLOAT - Heat loss due to radiation

#### 3.2.5 Metadata
- `Sheet_Source`: TEXT - Source sheet identifier (BOLR 1, BOL2, BOLR 3)

### 3.3 Table Statistics

| Table Name | Row Count | Column Count | Primary Key |
|------------|-----------|--------------|------------|
| boiler_data | 1098 | 110+ | None |
| boiler_bolr_1 | 366 | 110+ | None |
| boiler_bol2 | 366 | 110+ | None |
| boiler_bolr_3 | 366 | 110+ | None |
| weekly_stats | 156 | 8 | None |
| monthly_stats | 36 | 8 | None |

---

## 4. LangChain Agent Implementation

### 4.1 Agent Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  User Question  │────▶│  LLM Processing │────▶│  SQL Generation │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Response       │◀────│  Result         │◀────│  Query Execution│
│  Generation     │     │  Processing     │     │                 │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### 4.2 Agent Tools

The LangChain agent uses the following tools from the SQLDatabaseToolkit:

1. **sql_db_list_tables**: Lists all tables in the database
   ```python
   # Example usage
   Action: sql_db_list_tables
   Action Input: ""
   Observation: boiler_data, boiler_bolr_1, boiler_bol2, boiler_bolr_3, weekly_stats, monthly_stats
   ```

2. **sql_db_schema**: Retrieves the schema for specified tables
   ```python
   # Example usage
   Action: sql_db_schema
   Action Input: boiler_bolr_1
   Observation: CREATE TABLE boiler_bolr_1 (
     "Date" DATETIME,
     "LDO Consumption(kl)" FLOAT,
     "Coal ConsumptionFeeder(MT)" BIGINT,
     ...
   )
   ```

3. **sql_db_query**: Executes SQL queries against the database
   ```python
   # Example usage
   Action: sql_db_query
   Action Input: SELECT "Coal ConsumptionKPI" FROM boiler_bolr_1 WHERE Date = '2024-04-01'
   Observation: [{'Coal ConsumptionKPI': 1296.01}]
   ```

4. **sql_db_query_checker**: Checks SQL queries for errors before execution
   ```python
   # Example usage
   Action: sql_db_query_checker
   Action Input: SELECT "Coal ConsumptionKPI" FROM boiler_bolr_1 WHERE Date = '2024-04-01'
   Observation: The query looks correct. It selects the Coal ConsumptionKPI column from the boiler_bolr_1 table where the Date is April 1, 2024.
   ```

### 4.3 React Agent Pattern

The React agent follows a Reasoning and Acting pattern:

1. **Thought**: The agent reasons about what tables and columns are needed
   ```
   Thought: I need to find the value of Coal ConsumptionKPI for Boiler 1 on April 1, 2024. Let me first check what tables are available in the database.
   ```

2. **Action**: The agent lists tables or retrieves schemas
   ```
   Action: sql_db_list_tables
   Action Input: ""
   ```

3. **Observation**: The agent observes the database structure
   ```
   Observation: boiler_data, boiler_bolr_1, boiler_bol2, boiler_bolr_3, weekly_stats, monthly_stats
   ```

4. **Thought**: The agent formulates a SQL query
   ```
   Thought: I see that there's a table called boiler_bolr_1 which should contain data for Boiler 1. Let me check its schema to confirm it has the Coal ConsumptionKPI column.
   ```

5. **Action**: The agent executes the query
   ```
   Action: sql_db_query
   Action Input: SELECT "Coal ConsumptionKPI" FROM boiler_bolr_1 WHERE Date = '2024-04-01'
   ```

6. **Observation**: The agent observes the query results
   ```
   Observation: [{'Coal ConsumptionKPI': 1296.01}]
   ```

7. **Thought**: The agent interprets the results
   ```
   Thought: I have found the value of Coal ConsumptionKPI for Boiler 1 on April 1, 2024. The value is 1296.01.
   ```

8. **Action**: The agent returns a final answer
   ```
   The value of Coal ConsumptionKPI for Boiler 1 on April 1, 2024 was 1296.01.
   ```

---

## 5. Streamlit UI Implementation

### 5.1 UI Components

```
┌─────────────────────────────────────────────────────────────┐
│ Text-to-SQL Agent 🔍                                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ Enter your natural language question 👇                     │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │                                                         │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ [Query Results Display Area]                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Sidebar Implementation

```python
db = connect_to_db()
tables = db.get_usable_table_names()
st.sidebar.write("📋 Tables in DB:")
for t in tables:
    st.sidebar.markdown(f"- `{t}`")
```

This code:
1. Connects to the database
2. Gets the list of tables
3. Displays the tables in the sidebar

### 5.3 Query Processing Implementation

```python
# Input query
user_query = st.text_input("Enter your natural language question 👇")

if user_query:
    with st.spinner("Generating SQL and fetching results..."):
        try:
            result = agent.run(user_query)
            st.success("Query executed successfully!")
            st.write("**Agent's Response:**")
            st.write(result)
        except Exception as e:
            st.error(f"❌ Error: {e}")
```

This code:
1. Gets the user's query from the text input field
2. Shows a spinner while processing the query
3. Runs the agent with the user's query
4. Displays the result or an error message

---

## 6. Query Processing Flow

### 6.1 Query Processing Sequence Diagram

```
┌─────┐          ┌─────────┐          ┌────────┐          ┌──────────┐
│User │          │Streamlit│          │LangChain│          │SQLite DB │
└──┬──┘          └────┬────┘          └────┬────┘          └────┬─────┘
   │                  │                    │                     │
   │ Enter Query      │                    │                     │
   │─────────────────>│                    │                     │
   │                  │                    │                     │
   │                  │ Run Agent          │                     │
   │                  │───────────────────>│                     │
   │                  │                    │                     │
   │                  │                    │ List Tables         │
   │                  │                    │────────────────────>│
   │                  │                    │                     │
   │                  │                    │ Tables List         │
   │                  │                    │<────────────────────│
   │                  │                    │                     │
   │                  │                    │ Get Schema          │
   │                  │                    │────────────────────>│
   │                  │                    │                     │
   │                  │                    │ Schema              │
   │                  │                    │<────────────────────│
   │                  │                    │                     │
   │                  │                    │ Execute Query       │
   │                  │                    │────────────────────>│
   │                  │                    │                     │
   │                  │                    │ Query Results       │
   │                  │                    │<────────────────────│
   │                  │                    │                     │
   │                  │ Agent Response     │                     │
   │                  │<───────────────────│                     │
   │                  │                    │                     │
   │ Display Results  │                    │                     │
   │<─────────────────│                    │                     │
   │                  │                    │                     │
┌──┴──┐          ┌────┴────┐          ┌────┴────┐          ┌────┴─────┐
│User │          │Streamlit│          │LangChain│          │SQLite DB │
└─────┘          └─────────┘          └─────────┘          └──────────┘
```

### 6.2 Example Query Processing

#### 6.2.1 User Query
```
What was the value of Coal ConsumptionKPI on date 4-1-24 for Boiler 1?
```

#### 6.2.2 Agent Reasoning
```
Thought: I need to find the value of Coal ConsumptionKPI for Boiler 1 on April 1, 2024. Let me first check what tables are available in the database.

Action: sql_db_list_tables
Action Input: ""

Observation: boiler_data, boiler_bolr_1, boiler_bol2, boiler_bolr_3, weekly_stats, monthly_stats

Thought: I see that there's a table called boiler_bolr_1 which should contain data for Boiler 1. Let me check its schema to confirm it has the Coal ConsumptionKPI column.

Action: sql_db_schema
Action Input: boiler_bolr_1

Observation: CREATE TABLE boiler_bolr_1 (
  "Date" DATETIME,
  "LDO Consumption(kl)" FLOAT,
  "Coal ConsumptionFeeder(MT)" BIGINT,
  "Coal ConsumptionKPI" FLOAT,
  ...
)

Thought: I can see that the boiler_bolr_1 table has a Date column and a Coal ConsumptionKPI column. The date format in the query is "4-1-24" which likely means April 1, 2024. Let me query for this date.

Action: sql_db_query
Action Input: SELECT "Coal ConsumptionKPI" FROM boiler_bolr_1 WHERE Date = '2024-04-01'

Observation: [{'Coal ConsumptionKPI': 1296.01}]

Thought: I have found the value of Coal ConsumptionKPI for Boiler 1 on April 1, 2024. The value is 1296.01.

Final Answer: The value of Coal ConsumptionKPI for Boiler 1 on April 1, 2024 was 1296.01.
```

#### 6.2.3 SQL Query Generated
```sql
SELECT "Coal ConsumptionKPI" FROM boiler_bolr_1 WHERE Date = '2024-04-01'
```

#### 6.2.4 Final Response
```
The value of Coal ConsumptionKPI for Boiler 1 on April 1, 2024 was 1296.01.
```

---

## 7. Error Handling Implementation

### 7.1 Database Connection Error Handling

The system implements robust error handling for database connections:

```python
def connect_to_db():
    possible_db_paths = [
        '../boiler_data_all_sheets.db',
        os.path.join('..', 'boiler_data_all_sheets.db'),
        os.path.abspath('boiler_data_all_sheets.db'),
        os.path.abspath(os.path.join('..', 'boiler_data_all_sheets.db')),
    ]
    for db_path in possible_db_paths:
        if os.path.exists(db_path):
            try:
                db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
                return db
            except Exception as e:
                st.error(f"Failed to connect to {db_path}: {e}")
    st.stop()
```

This function:
1. Tries multiple possible paths to find the database
2. Checks if each path exists before attempting to connect
3. Catches and displays any connection errors
4. Stops the application if no valid database connection can be established

### 7.2 Query Execution Error Handling

The system implements error handling for query execution:

```python
try:
    result = agent.run(user_query)
    st.success("Query executed successfully!")
    st.write("**Agent's Response:**")
    st.write(result)
except Exception as e:
    st.error(f"❌ Error: {e}")
```

This code:
1. Tries to run the agent with the user's query
2. Displays a success message and the result if successful
3. Catches and displays any errors that occur during query execution

### 7.3 Agent Self-Correction

The LangChain agent implements self-correction for SQL queries:

```python
system_prompt = """
...
You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.
...
"""
```

This instruction in the system prompt ensures that:
1. The agent checks its SQL queries before execution
2. If a query fails, the agent rewrites it and tries again
3. The agent continues this process until it succeeds or reaches a maximum number of attempts

### 7.4 Date Format Handling

The agent handles various date formats in user queries:

```python
# Example query with ambiguous date format
question = "What was the value of Coal ConsumptionKPI on date 4-1-24 for Boiler 1?"

# Agent reasoning
Thought: The date format in the query is "4-1-24" which likely means April 1, 2024. Let me query for this date.

# SQL query with standardized date format
Action Input: SELECT "Coal ConsumptionKPI" FROM boiler_bolr_1 WHERE Date = '2024-04-01'
```

The agent:
1. Recognizes the ambiguous date format in the user's query
2. Interprets it as April 1, 2024
3. Converts it to the standard SQL date format (YYYY-MM-DD)

### 7.5 Out-of-Scope Query Handling

The agent handles queries outside the domain of boiler data:

```python
# Example out-of-scope query
question = "who is the richest person in the world?"

# Agent response
I don't have information about who the richest person in the world is in the database. The database contains information about boiler performance data including metrics like coal consumption, unit generation, boiler efficiency, and emissions data for three boilers (BOLR 1, BOL2, and BOLR 3). I can help you query this boiler-related data instead.
```

The agent:
1. Recognizes that the query is outside the scope of the boiler database
2. Explains that it can only answer questions related to the boiler data
3. Suggests the types of queries it can help with

---

## 8. Performance Optimization Details

### 8.1 Database Optimizations

#### 8.1.1 Pre-aggregated Tables

The system uses pre-aggregated tables to improve query performance:

```python
# Weekly aggregation
weekly_agg = combined_df.groupby(['Sheet_Source', 'Week']).agg({
    'Coal ConsumptionFeeder(MT)': 'sum',
    'Unit Generation': 'sum',
    'Boiler Efficiency': 'mean',
    'SOx (mg/m3)': 'mean',
    'NOx (mg/m3)': 'mean',
    'CO (PPM)': 'mean'
}).reset_index()

# Monthly aggregation
monthly_agg = combined_df.groupby(['Sheet_Source', 'Month']).agg({
    'Coal ConsumptionFeeder(MT)': 'sum',
    'Unit Generation': 'sum',
    'Boiler Efficiency': 'mean',
    'SOx (mg/m3)': 'mean',
    'NOx (mg/m3)': 'mean',
    'CO (PPM)': 'mean'
}).reset_index()
```

Benefits:
1. Reduces the need for expensive aggregation operations at query time
2. Improves response time for common time-based queries
3. Simplifies the SQL queries generated by the agent

#### 8.1.2 Selective Column Querying

The agent is instructed to query only the necessary columns:

```python
system_prompt = """
...
Never query for all the columns from a specific table,
only ask for the relevant columns given the question.
...
"""
```

Benefits:
1. Reduces data transfer between the database and application
2. Decreases memory usage for query results
3. Improves query execution time

### 8.2 LLM Optimizations

#### 8.2.1 Model Selection

The system uses GPT-4.1 with a temperature of 0:

```python
llm = ChatOpenAI(model="gpt-4.1", temperature=0)
```

Benefits:
1. GPT-4.1 provides superior performance for complex reasoning tasks
2. Temperature of 0 ensures deterministic outputs, which is important for SQL generation
3. Reduces the need for multiple query attempts

#### 8.2.2 Prompt Engineering

The system uses a carefully crafted system prompt:

```python
system_prompt = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.
...
"""
```

Benefits:
1. Provides clear instructions for the agent's behavior
2. Includes specific guidelines for SQL query generation
3. Reduces the number of iterations needed to generate a correct query

### 8.3 UI Performance Optimizations

#### 8.3.1 Asynchronous Processing

The system uses a spinner to indicate processing:

```python
with st.spinner("Generating SQL and fetching results..."):
    try:
        result = agent.run(user_query)
        st.success("Query executed successfully!")
        st.write("**Agent's Response:**")
        st.write(result)
    except Exception as e:
        st.error(f"❌ Error: {e}")
```

Benefits:
1. Provides feedback to the user during potentially long-running operations
2. Prevents the UI from appearing frozen during processing
3. Improves the user experience

#### 8.3.2 Caching Implementation

Streamlit's built-in caching can be used to improve performance:

```python
@st.cache_data
def run_query(query):
    return agent.run(query)
```

Benefits:
1. Caches the results of identical queries
2. Reduces processing time for repeated queries
3. Decreases API costs for LLM calls

### 8.4 Scalability Considerations

For larger datasets or higher query volumes, consider:

#### 8.4.1 Database Indexing

```sql
CREATE INDEX idx_date ON boiler_bolr_1(Date);
CREATE INDEX idx_sheet_source ON boiler_data(Sheet_Source);
```

Benefits:
1. Speeds up queries that filter or sort by indexed columns
2. Improves performance for large tables
3. Reduces database load

#### 8.4.2 Connection Pooling

```python
from sqlalchemy.pool import QueuePool

engine = create_engine(
    f'sqlite:///{db_path}',
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10
)
```

Benefits:
1. Reuses database connections instead of creating new ones
2. Reduces connection overhead
3. Improves performance for multiple concurrent users

#### 8.4.3 Load Balancing

For a production deployment, consider:
1. Deploying multiple instances of the application
2. Using a load balancer to distribute requests
3. Implementing a shared cache for query results

#### 8.4.4 Caching Layer

```python
import redis

cache = redis.Redis(host='localhost', port=6379, db=0)

def get_cached_result(query):
    cached = cache.get(query)
    if cached:
        return json.loads(cached)
    return None

def set_cached_result(query, result):
    cache.set(query, json.dumps(result), ex=3600)  # Cache for 1 hour
```

Benefits:
1. Reduces database and LLM load for repeated queries
2. Improves response time for common queries
3. Can be shared across multiple application instances
