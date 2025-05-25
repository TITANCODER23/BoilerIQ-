# BoilerIQ Documentation Guide

## Overview

This guide provides an overview of the BoilerIQ GPT-powered boiler data query system and the documentation available for understanding, maintaining, and extending the system.

## What is BoilerIQ?

BoilerIQ is a natural language interface for querying boiler operational data. The system allows users to ask questions in plain English about boiler performance, efficiency, emissions, and other operational metrics. These natural language questions are automatically converted into SQL queries, executed against a structured database, and the results are presented to the user in an easy-to-understand format.

## Key Features

- **Natural Language Interface**: Query complex boiler data using everyday language
- **Multi-Boiler Analysis**: Compare performance across multiple boilers (BOLR 1, BOL2, BOLR 3)
- **Temporal Analysis**: Analyze data across different time periods (daily, weekly, monthly)
- **Comprehensive Data Coverage**: Access over 100 operational parameters and metrics
- **User-Friendly Web Interface**: Simple, intuitive interface built with Streamlit

## Documentation Structure

The BoilerIQ documentation is organized into three main documents:

### 1. [BoilerIQ Technical Documentation](./BoilerIQ_Technical_Documentation.md)

This is the primary technical documentation that provides a comprehensive overview of the BoilerIQ system. It covers:

- Project overview and objectives
- System architecture
- Data pipeline architecture and flow
- Database schema and table structures
- Data preprocessing methodology
- Natural language to SQL query translation
- Frontend/interface specifications
- Deployment and setup instructions
- Error handling and edge cases
- Performance considerations and optimizations
- Implementation roadmap

**Audience**: Technical team members, developers, data engineers, and system administrators who need to understand the overall system architecture and functionality.

### 2. [BoilerIQ Implementation Details](./BoilerIQ_Implementation_Details.md)

This document provides in-depth technical details about the implementation of the BoilerIQ system. It includes:

- Detailed code walkthroughs
- Data pipeline implementation specifics
- Database schema details
- LangChain agent implementation
- Streamlit UI implementation
- Query processing flow
- Error handling implementation
- Performance optimization details

**Audience**: Developers who need to maintain, modify, or extend the system's functionality.

### 3. [Boiler Database Creation Process](./boiler_database_creation_process.txt)

This document explains the detailed process of how the SQLite database was created, including:

- Overview of the data source
- Data loading process
- Data cleaning and preprocessing steps
- Feature engineering
- Data aggregation
- Database structure and tables

**Audience**: Data engineers and analysts who need to understand the data pipeline and database structure.

## Key System Components

### Data Pipeline

The data pipeline processes raw boiler data from Excel files, performs cleaning and preprocessing, and loads the data into a structured SQLite database. The pipeline includes:

- Data loading from Excel sheets
- Data cleaning and standardization
- Missing value handling
- Outlier detection and handling
- Feature engineering
- Data aggregation
- Database storage

### Database

The SQLite database stores the processed boiler data in a structured format with multiple tables:

- `boiler_data`: Combined data from all boilers
- `boiler_bolr_1`, `boiler_bol2`, `boiler_bolr_3`: Individual boiler data
- `weekly_stats`, `monthly_stats`: Aggregated statistics

### LangChain Agent

The LangChain agent converts natural language questions into SQL queries using:

- OpenAI's GPT-4.1 model
- React (Reasoning and Acting) architecture
- SQL database toolkit
- Custom system prompt

### Streamlit UI

The Streamlit web interface provides:

- Simple text input for natural language questions
- Display of query results
- Error handling and feedback
- Database metadata in the sidebar

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- OpenAI API key

### Setup

1. Clone the repository
2. Create a virtual environment
3. Install dependencies from `requirements.txt`
4. Set up environment variables (OpenAI API key)
5. Run the Streamlit application

### Example Queries

- "What was the value of Coal ConsumptionKPI on date 4-1-24 for Boiler 1?"
- "What is the average boiler efficiency for each boiler in March 2024?"
- "Which boiler had the highest NOx emissions last week?"
- "Compare the Unit Generation between all boilers for the month of January"
- "What is the correlation between Coal Consumption and Boiler Efficiency for Boiler 2?"

## Implementation Roadmap

The BoilerIQ system is being developed in phases:

1. **Phase 1 (Completed)**: Core functionality
   - Data preprocessing pipeline
   - SQLite database setup
   - LangChain agent for text-to-SQL conversion
   - Basic Streamlit UI

2. **Phase 2**: Enhanced features
   - Advanced data visualization
   - Support for more complex queries
   - Improved error handling
   - Query history and favorites

3. **Phase 3**: Production readiness
   - Comprehensive testing
   - Performance optimizations
   - User authentication
   - Deployment automation

4. **Phase 4**: Advanced capabilities
   - Predictive analytics
   - Anomaly detection
   - Maintenance recommendations
   - Integration with other plant systems

## Conclusion

The BoilerIQ system provides a powerful natural language interface for querying complex boiler operational data. The comprehensive documentation provided in this repository will help you understand, maintain, and extend the system to meet your specific needs.

For detailed information, please refer to the individual documentation files linked above.
