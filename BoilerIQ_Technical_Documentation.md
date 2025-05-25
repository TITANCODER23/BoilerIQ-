# BoilerIQ Technical Documentation

## 1. System Overview
BoilerIQ is a GPT-powered boiler data query system that allows users to ask natural language questions about boiler performance metrics and get accurate SQL query results. The system consists of:
- SQLite database containing boiler performance data
- Streamlit web interface
- LangChain agent for natural language to SQL translation

## 2. Database Architecture

### 2.1 Database Schema
The database contains 6 tables with detailed boiler performance metrics:
- `boiler_data` (1098 rows) - Combined data from all boilers
- `boiler_bolr_1` (366 rows) - Boiler 1 data
- `boiler_bol2` (366 rows) - Boiler 2 data
- `boiler_bolr_3` (366 rows) - Boiler 3 data
- `weekly_stats` (156 rows) - Weekly aggregated data
- `monthly_stats` (36 rows) - Monthly aggregated data

### 2.2 Data Pipeline
1. Data loaded from Excel sheets
2. Cleaning and preprocessing:
   - Column name standardization
   - Missing value handling (median imputation)
   - Outlier detection and handling (IQR method)
3. Feature engineering:
   - Time-based features (Year, Month, Week, Day, DayOfWeek)
   - Sheet source tracking
4. Data aggregation:
   - Weekly statistics
   - Monthly statistics
5. Database export

## 3. Application Architecture

### 3.1 Streamlit UI (app.py)
- Provides web interface for user queries
- Connects to SQLite database
- Initializes LangChain agent
- Displays query results

### 3.2 LangChain Agent (agent_texttosql_main.ipynb)
- Uses GPT-4 for natural language understanding
- Custom system prompt for SQL generation:
  ```text
  You are an agent designed to interact with a SQL database...
  Always limit your query to at most 5 results...
  Double check your query before executing...
  ```
- Tools:
  - Table listing
  - Schema inspection
  - Query checking
  - Query execution

## 4. Implementation Details

### 4.1 Database Connection
- Multiple path fallbacks for database location
- SQLAlchemy connection pooling
- Automatic schema inspection

### 4.2 Query Processing
1. User enters natural language question
2. Agent:
   - Identifies relevant tables
   - Inspects table schemas
   - Generates SQL query
   - Validates query syntax
   - Executes query
   - Formats results

### 4.3 Error Handling
- Invalid query detection
- Query rewriting on failure
- User-friendly error messages

## 5. Deployment Instructions

### 5.1 Requirements
- Python 3.9+
- Required packages:
  ```text
  streamlit
  langchain
  sqlalchemy
  pandas
  python-dotenv
  ```

### 5.2 Setup
1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set environment variables:
   ```text
   OPENAI_API_KEY=your_key
   LANGCHAIN_API_KEY=your_key
   ```
4. Run application: `streamlit run app.py`

## 6. Example Queries
1. "What was the value of Coal ConsumptionKPI on date 4-1-24 for Boiler 1?"
2. "Show weekly averages of Boiler Efficiency for Boiler 2"
3. "Compare monthly SOx emissions between all boilers"

## 7. Performance Considerations
- Query optimization through GPT-4 validation
- Result limiting (max 5 rows by default)
- Cached database connections
- Efficient data types in schema

## 8. Limitations
- Only supports SELECT queries
- Limited to 5 results by default
- Requires precise date formatting
- Complex joins may fail

## 9. Future Enhancements
- Support for more complex queries
- Visualization of results
- User-defined query limits
- Additional data sources
