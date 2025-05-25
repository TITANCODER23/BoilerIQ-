# BoilerIQ - Text-to-SQL Analytics Platform

![BoilerIQ Logo](https://via.placeholder.com/150x50?text=BoilerIQ+Logo)

BoilerIQ is an intelligent analytics platform that transforms natural language questions about boiler performance data into SQL queries, providing instant insights into operational metrics, efficiency, and emissions.

## Features

- **Natural Language to SQL**: Ask questions in plain English and get accurate SQL results
- **Multi-Boiler Analysis**: Compare performance across different boiler units
- **Time-Series Insights**: Analyze trends over days, weeks, and months
- **Efficiency Metrics**: Track boiler efficiency and identify improvement areas
- **Emissions Monitoring**: Monitor SOx, NOx, and CO emissions
- **Interactive UI**: User-friendly Streamlit interface for easy exploration

## Technology Stack

- **Backend**: Python, SQLite
- **Data Processing**: Pandas, NumPy
- **Natural Language Processing**: LangChain, OpenAI GPT-4
- **UI**: Streamlit
- **Visualization**: Matplotlib, Plotly (optional)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/BoilerIQ.git
cd BoilerIQ
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
```
Edit the .env file with your OpenAI API key and other configurations.

4. Run the application:
```bash
streamlit run app.py
```

## File Structure

```
BoilerIQ/
├── Data Cleaning and Preprocessing/    # Cleaned data files
├── Gen_ai/                             # AI agent implementation
│   ├── agent_texttosql_main.ipynb      # LangChain agent notebook
│   ├── app.py                          # Streamlit application
│   └── analyse_db_schema.py            # Database schema analysis
├── boiler_data_all_sheets.db           # SQLite database
├── boiler_database_creation_process.txt # DB creation documentation
├── BoilerIQ_Documentation_Guide.md     # Documentation guide
├── BoilerIQ_Implementation_Details.md  # Technical implementation
├── BoilerIQ_Technical_Documentation.md # System documentation
├── data_analysis_updated.ipynb         # Data analysis notebook
├── data_preprocessing_all_sheets.py    # Data cleaning script
├── db_schema.json                      # Database schema
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
└── .env.example                        # Environment template
```

## Usage Examples

1. **Basic Query**:
```
"What was the coal consumption for Boiler 1 last week?"
```

2. **Comparative Analysis**:
```
"Compare the efficiency of all boilers in January 2024"
```

3. **Trend Analysis**:
```
"Show me the NOx emissions trend for Boiler 2 over the past 6 months"
```

## Screenshots

![Query Interface](https://via.placeholder.com/600x400?text=Streamlit+UI)
*Natural language query interface*

![Results Display](https://via.placeholder.com/600x400?text=Query+Results)
*Query results visualization*

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Project Maintainer: [Your Name]  
Email: your.email@example.com  
Project Link: [https://github.com/yourusername/BoilerIQ](https://github.com/yourusername/BoilerIQ)
