import os
import pathlib
import requests

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

# ---------------------------------------------------
# 1. SET API KEY
# ---------------------------------------------------

os.environ["GOOGLE_API_KEY"] = str(os.getenv("GOOGLE_API_KEY"))
print("GOOGLE_API_KEY set:", "GOOGLE_API_KEY" in os.environ)
# ---------------------------------------------------
# 2. INITIALIZE GEMINI MODEL
# ---------------------------------------------------

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# ---------------------------------------------------
# 3. DOWNLOAD CHINOOK DATABASE
# ---------------------------------------------------

url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
local_path = pathlib.Path("Chinook.db")

if local_path.exists():
    print("Database already exists.")
else:
    print("Downloading database...")
    response = requests.get(url)

    if response.status_code == 200:
        local_path.write_bytes(response.content)
        print("Database downloaded successfully.")
    else:
        print("Download failed.")

# ---------------------------------------------------
# 4. CONNECT TO DATABASE
# ---------------------------------------------------

db = SQLDatabase.from_uri("sqlite:///Chinook.db")

print("Database Dialect:", db.dialect)
print("Available Tables:", db.get_usable_table_names())

# ---------------------------------------------------
# 5. CREATE SQL TOOLS
# ---------------------------------------------------

toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

print("\nAvailable Tools:\n")

for tool in tools:
    print(f"{tool.name}: {tool.description}\n")

# ---------------------------------------------------
# 6. SYSTEM PROMPT FOR AGENT
# ---------------------------------------------------

system_prompt = f"""
You are an agent designed to interact with a SQL database.

Given an input question, create a syntactically correct {db.dialect}
query to run, then look at the results of the query and return the answer.

Unless the user specifies otherwise, limit results to 5 rows.

Always follow these steps:

1. Look at available tables.
2. Identify relevant tables.
3. Check their schema.
4. Write a SQL query.
5. Double check query before executing.

Rules:
- NEVER run INSERT, UPDATE, DELETE, DROP.
- Only retrieve relevant columns.
- Fix errors if query fails.
"""

# ---------------------------------------------------
# 7. CREATE AGENT
# ---------------------------------------------------

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt
)

print("\nSQL Agent Ready!\n")

# ---------------------------------------------------
# 8. RUN INTERACTIVE LOOP
# ---------------------------------------------------

while True:

    question = input("\nAsk a question about the database (or type exit): ")

    if question.lower() == "exit":
        break

    for step in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()