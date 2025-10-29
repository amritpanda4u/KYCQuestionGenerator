# ImportJson.py
import os
import json
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from llama_cpp import Llama
from langchain_core.documents import Document

# -----------------------------
# Config paths
# -----------------------------
CHROMA_DB_PATH = "/Users/amrit/chroma_kb"  # path to your existing vector store
LLAMA_MODEL_PATH = "/Users/amrit/models/llama2-7b-chat-gguf/llama-2-7b-chat.Q4_K_M.gguf"
NEW_CUSTOMERS_JSON = "/Users/amrit/Desktop/Amrit/BITS/S4/new_customers.json"


# ----------------- LOAD EMBEDDINGS -----------------
print("🔹 Loading local embeddings model (sentence-transformers)...")
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ----------------- CONNECT TO CHROMA -----------------
print("🔹 Connecting to existing Chroma vector database...")
vectordb = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=emb)

# ----------------- LOAD NEW CUSTOMER DATA -----------------
print(f"🔹 Loading new customer data from {NEW_CUSTOMERS_JSON}...")
with open(NEW_CUSTOMERS_JSON, "r", encoding="utf-8") as f:
    new_customers = json.load(f)

# ----------------- PREPARE DOCUMENTS -----------------
documents = []
for cust in new_customers:
    cust_id = cust.get("customerId", "unknown")

    # Flatten KYC fields into a single string
    text_fields = [
        cust.get("name", ""),
        str(cust.get("age", "")),
        cust.get("address", ""),
        cust.get("phone", ""),
        cust.get("email", ""),
        json.dumps(cust.get("FATCA", {})),
        json.dumps(cust.get("CRS", {})),
        json.dumps(cust.get("FEC", {})),
    ]
    text = " | ".join(filter(None, text_fields))

    if text.strip():
        doc = Document(page_content=text, metadata={"customerId": cust_id})
        documents.append(doc)

print(f"🔹 Adding {len(documents)} new documents to Chroma DB...")
if documents:
    vectordb.add_documents(documents)
    vectordb.persist()
    print("✅ New customer embeddings added successfully.")
else:
    print("⚠️ No new documents to add.")

