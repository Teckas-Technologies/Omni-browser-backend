import os
from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langgraph.graph import StateGraph, END
import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# === Define RAG State Schema ===
class RAGState(TypedDict):
    query: str
    docs: List[Document]
    answer: str

# === Load and index documents ===
def load_and_index_documents():
    print("📄 Loading PDF and creating FAISS index...")
    loader = PyPDFLoader("sample.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("faiss_index")
    print("✅ FAISS index created and saved.")

# === Build LangGraph ===

# Ensure FAISS index exists
if not os.path.exists("faiss_index"):
    load_and_index_documents()

# Load retriever
retriever = FAISS.load_local(
    "faiss_index",
    OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")),
    allow_dangerous_deserialization=True
)

# Node functions
def retrieve(state: RAGState) -> RAGState:
    query = state["query"]
    docs = retriever.similarity_search(query, k=3)
    return {"query": query, "docs": docs}

def generate(state: RAGState) -> RAGState:
    docs_text = "\n".join([doc.page_content for doc in state["docs"]])
    prompt = f"Answer the following based on the documents:\n\n{docs_text}\n\nQuestion: {state['query']}"
    llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
    answer = llm.invoke(prompt)
    return {"query": state["query"], "docs": state["docs"], "answer": answer}

# Build graph
builder = StateGraph(RAGState)
builder.add_node("retriever", retrieve)
builder.add_node("generator", generate)
builder.set_entry_point("retriever")
builder.add_edge("retriever", "generator")
builder.add_edge("generator", END)

graph = builder.compile()

# ✅ Expose this for FastAPI
def get_answer_from_graph(query: str) -> str:
    result = graph.invoke({"query": query})
    return result["answer"]

# Optional CLI
if __name__ == "__main__":
    print("🤖 Ready! Ask questions about your PDF.")
    while True:
        question = input("\nAsk a question: ")
        result = get_answer_from_graph(question)
        print("\n🧠 Answer:", result)
