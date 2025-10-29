import os
import json
import sys
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp

# ============================================================
# ✅ CONFIGURATION
# ============================================================

CUSTOMER_DATA_PATH = "/Users/amrit/Desktop/Amrit/BITS/S4/CustomerData.json"
NEW_CUSTOMERS_JSON = "/Users/amrit/Desktop/Amrit/BITS/S4/new_customers.json"
TEMPLATES_PATH = "/Users/amrit/Desktop/Amrit/BITS/S4/OutreachTemplates.json"
CHROMA_DB_DIR = "/Users/amrit/Desktop/Amrit/BITS/S4/ChromaDB"
MODEL_PATH = "/Users/amrit/models/llama2-7b-chat-gguf/llama-2-7b-chat.Q4_K_M.gguf"
OUTPUT_PATH = "/Users/amrit/Desktop/Amrit/BITS/S4/generated_outreach.json"

TEST_MODE = False   # 🧪 Only generate for first customer
MAX_SUMMARY_LEN = 1500  # limit input to prevent llama_decode overflow

# ============================================================
# ✅ HELPER FUNCTIONS
# ============================================================

def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return []

def save_json(filepath, data):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

# ============================================================
# ✅ EMBEDDINGS & VECTORSTORE
# ============================================================

print("🔹 Loading local embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("🔹 Connecting to local Chroma DB...")
vectordb = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# ============================================================
# ✅ LOAD DATA
# ============================================================

existing_customers = load_json(CUSTOMER_DATA_PATH)
new_customers = load_json(NEW_CUSTOMERS_JSON)
templates = load_json(TEMPLATES_PATH)

existing_ids = {cust.get("customerId") for cust in existing_customers}
new_entries = [cust for cust in new_customers if cust.get("customerId") not in existing_ids]

print(f"✅ Loaded {len(existing_customers)} existing customers, {len(new_entries)} new customers, and {len(templates)} outreach templates.")

if not new_entries:
    print("ℹ️ No new customers found — skipping outreach generation.")
    sys.exit(0)

# ============================================================
# ✅ EMBED OUTREACH TEMPLATES
# ============================================================

if templates:
    print("🔹 Embedding outreach templates into Chroma DB...")
    texts = [t.get("template_text", "") for t in templates if "template_text" in t]
    metadatas = [{"template_id": t.get("id", f"template_{i}")} for i, t in enumerate(templates)]
    vectordb.add_texts(texts=texts, metadatas=metadatas)
    vectordb.persist()
    print(f"✅ Embedded {len(texts)} outreach templates into Chroma DB.")
else:
    print("⚠️ No valid templates found for embedding — please check JSON structure.")

# ============================================================
# ✅ LOAD LOCAL LLAMA MODEL (Optimized for Mac Stability)
# ============================================================

print("🔹 Loading local LLaMA model...")
llm = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.3,
    max_tokens=256,
    n_ctx=2048,        # reduced for stability
    n_threads=6,
    n_gpu_layers=16,   # tweak: reduce if memory is tight, set 0 for CPU mode
    verbose=False
)

# ============================================================
# ✅ RAG PROMPT TEMPLATE
# ============================================================

prompt_template = """
You are a compliance-focused outreach specialist.
Use the context below to create a concise, compliant, and personalized outreach message
for the given KYC customer profile.

Context:
{context}

Question:
{question}

Guidelines:
- Keep tone professional, warm, and compliant.
- Avoid generic filler language.
- Summarize only the relevant background and offer specific guidance.

Final Message:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# ============================================================
# ✅ GENERATE OUTREACH MESSAGES (Only for New Customers)
# ============================================================

print(f"🧪 TEST_MODE {'ENABLED' if TEST_MODE else 'DISABLED'} → Generating outreach for {1 if TEST_MODE else len(new_entries)} new customer(s).")

results = []
for i, customer in enumerate(new_entries[:1 if TEST_MODE else len(new_entries)]):
    cust_id = customer.get("customerId", f"CUST_NEW_{i+1}")
    name = customer.get("name", "Customer")
    summary = json.dumps(customer, indent=2)

    # ✂️ Trim overly long input to prevent llama_decode crash
    trimmed_summary = summary[:MAX_SUMMARY_LEN] + ("..." if len(summary) > MAX_SUMMARY_LEN else "")

    print(f"\n🧩 Generating outreach for {cust_id} - {name}...")
    query = f"Generate a personalized outreach message for this KYC profile:\n{trimmed_summary}"

    try:
        retrieved_docs = retriever.get_relevant_documents(query)
        print(f"🔍 Retrieved {len(retrieved_docs)} relevant docs for {cust_id}.")

        result = qa_chain.invoke({"query": query})
        outreach_msg = result.get("result", "⚠️ No response generated.")

        results.append({
            "customer_id": cust_id,
            "name": name,
            "outreach_message": outreach_msg
        })

        print(f"✅ Generated outreach for {cust_id} ({name}): {outreach_msg[:100]}...")

    except Exception as e:
        print(f"❌ Error generating outreach for {cust_id}: {str(e)}")
        results.append({
            "customer_id": cust_id,
            "name": name,
            "error": str(e)
        })

# ============================================================
# ✅ SAVE OUTPUT & UPDATE MASTER CUSTOMER FILE
# ============================================================

save_json(OUTPUT_PATH, results)
print(f"\n📄 Outreach messages saved → {OUTPUT_PATH}")

# Append new customers to main data file
updated_customers = existing_customers + new_entries
save_json(CUSTOMER_DATA_PATH, updated_customers)
print(f"✅ Updated {CUSTOMER_DATA_PATH} with {len(new_entries)} new customers.")
print("✅ Pipeline completed successfully.")
