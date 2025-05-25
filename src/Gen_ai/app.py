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