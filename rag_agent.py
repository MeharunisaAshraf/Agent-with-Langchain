# ⚠️ FOR LOCAL TESTING ONLY — DO NOT COMMIT THIS FILE

import os
import bs4
import faiss

# Temporary manual key (replace with your NEW key locally only)
os.environ["GOOGLE_API_KEY"] = "AIzaSyAOA0UilC8opHWnjz6Vh7IBk_E2ezbTY5c"

os.environ["USER_AGENT"] = "rag-agent"

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.tools import tool
from langchain.agents import create_agent

# Load blog
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
)
docs = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
splits = text_splitter.split_documents(docs)

# Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

embedding_dim = len(embeddings.embed_query("hello world"))
index = faiss.IndexFlatL2(embedding_dim)

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

vector_store.add_documents(splits)

@tool
def retrieve_context(query: str) -> str:
    """Retrieve relevant context from the vector store for a given query."""
    docs = vector_store.similarity_search(query, k=2)
    return "\n\n".join(doc.page_content for doc in docs)

# Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

agent = create_agent(
    model=model,
    tools=[retrieve_context],
    system_prompt="You are a helpful assistant. Use retrieval tool when needed."
)

print("Gemini RAG Agent Ready!")

while True:
    query = input("Ask a question (type 'exit'): ")
    if query.lower() == "exit":
        break

    response = agent.invoke(
        {"messages": [{"role": "user", "content": query}]}
    )

    print(response["messages"][-1].content)