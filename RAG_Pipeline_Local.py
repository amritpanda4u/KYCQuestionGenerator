"""
RAG_Pipeline_Local.py
---------------------------------------------------------
Local Retrieval-Augmented Generation (RAG) System
for KYC Client Outreach Question Generation
---------------------------------------------------------
Recommended versions:
  pip install -U langchain>=0.3.7 langchain-core>=0.3.7 \
      langchain-community>=0.3.7 langchain-chroma>=0.1.7 \
      langchain-huggingface>=0.1.4 chromadb sentence-transformers \
      llama-cpp-python
"""

import os
import json
from typing import Dict

# ✅ LangChain modular ecosystem (v0.3.7+)
from langchain.chains import RetrievalQA        # <-- correct path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_core.documents import Document

# ------------------------------------------------------------
# 1️⃣ CONFIGURATION
# ------------------------------------------------------------

MODEL_PATH = "/Users/amrit/models/llama2-7b-chat-gguf/llama-2-7b-chat.Q4_K_M.gguf"
CHROMA_DB_PATH = "/Users/amrit/chroma_kb"
OUTREACH_JSON_PATH = "/Users/amrit/Desktop/Amrit/BITS/S4/Output.json"
CUSTOMER_JSON_PATH = "/Users/amrit/Desktop/Amrit/BITS/S4/CustomerData.json"

# ------------------------------------------------------------
# 2️⃣ LOAD LOCAL EMBEDDINGS AND VECTOR STORE
# ------------------------------------------------------------

print("🔹 Loading local embeddings model (sentence-transformers)...")
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("🔹 Connecting to local Chroma vector database...")
vectordb = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=emb)

# ------------------------------------------------------------
# 3️⃣ INITIALIZE LOCAL LLAMA MODEL (GPU via Metal)
# ------------------------------------------------------------

print("🔹 Loading local LLaMA 2 model...")
llm = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.2,
    max_tokens=512,
    n_ctx=2048,
    n_threads=6,         # M1 efficiency + performance cores
    n_gpu_layers=32,     # Run most layers on GPU (Metal)
    verbose=False
)

# ------------------------------------------------------------
# 4️⃣ DEFINE HELPER FUNCTIONS
# ------------------------------------------------------------

def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, list):
            return {item.get('customerId', f'unknown_{i}'): item for i, item in enumerate(data)}
        return data


def generate_outreach(customer_id: str, customers: dict, qa_chain) -> str:
    """Generate a contextual outreach message for a given customer."""
    cust = customers.get(customer_id)
    if not cust:
        raise ValueError(f"Customer {customer_id} not found.")

    # Create a KYC summary for retrieval
    cust_summary = f"""
    Customer: {cust.get('name')}
    FATCA: {json.dumps(cust.get('FATCA'), indent=2)}
    CRS: {json.dumps(cust.get('CRS'), indent=2)}
    FEC: {json.dumps(cust.get('FEC'), indent=2)}
    """

    query = f"Generate an outreach message based on KYC profile:\n{cust_summary}"
    result = qa_chain.invoke({"query": query})   # updated .invoke API
    if isinstance(result, dict) and "result" in result:
        return result["result"].strip()
    return str(result).strip()

# ------------------------------------------------------------
# 5️⃣ LOAD DATA
# ------------------------------------------------------------

print("🔹 Loading KYC and Outreach data...")
outreach_kb = load_json(OUTREACH_JSON_PATH)
customers = load_json(CUSTOMER_JSON_PATH)
print(f"✅ Loaded {len(customers)} customers and {len(outreach_kb)} outreach templates.")

# ------------------------------------------------------------
# 6️⃣ BUILD RAG CHAIN
# ------------------------------------------------------------

print("🔹 Building Retrieval-Augmented QA chain...")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
)

# ------------------------------------------------------------
# 7️⃣ RUN A SAMPLE QUERY
# ------------------------------------------------------------

customer_id = "CUST011"  # Example: US Person + Gambling source
print(f"\n🧩 Generating outreach for {customer_id}...\n")

try:
    outreach_msg = generate_outreach(customer_id, customers, qa_chain)
    print("=== Generated Outreach Message ===\n")
    print(outreach_msg)
except Exception as e:
    print("⚠️ Error generating outreach:", e)

print("\n✅ Pipeline completed successfully.")

