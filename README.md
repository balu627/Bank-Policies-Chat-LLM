**🏦 Bank Policy Assistant – AI-Powered Policy Q&A (Learning Project)**

This project is part of my personal learning journey into AI, LLMs, embeddings, vector databases, and full-stack app building.
It is not a production system, but a structured, end-to-end hands-on project to understand:

PDF ingestion & chunking

Embedding generation

FAISS vector search

Using Gemini LLMs with structured JSON outputs

Retrieval-Augmented Generation (RAG)

Session-based chat

Streamlit UI + FastAPI backend

Bank policy domain logic

**📌 Project Summary**

This application lets a user ask banking queries such as:

“How to apply for a credit card at SBI?”

“What are the eligibility conditions for HDFC loan?”

“What is the account opening process?”

The system does RAG retrieval from bank policy PDF documents and generates an AI answer with three structured sections:
✅ Section 1A — Summary (from policy documents only)

Uses only the retrieved PDF chunks.
no external knowledge.

✅ Section 1B — Step-wise Process (from policy documents only)

Detailed procedural steps if the policy mentions them.
If not mentioned → the model explicitly says so.

✅ Section 2 — Sources

Shows:

Bank name

Document name

Clean snippet (AI fixes broken chunk boundaries)

Example:

Bank: SBI
Document: sbi-policy-for-the-issuance-and-conduct-of-credit-cards-2023.pdf
Snippet: "Credit Cards can be issued to MSMEs... assessed based on financial statements..."

✅ Section 3 — Cost Saving Tips (general online info)

This section can use online/general banking knowledge,
but must state clearly:

“This section is based on general/online information and not from the policy documents.”

🧠 Architecture Overview
                        ┌────────────────────────┐
                        │        UI Layer        │
                        │     (Streamlit App)    │
                        └────────────┬───────────┘
                                     │
                                     ▼
                        ┌────────────────────────┐
                        │     FastAPI Backend    │
                        ├────────────────────────┤
                        │  /ask endpoint         │
                        │  Session store         │
                        │  RAG pipeline          │
                        └────────────┬───────────┘
                                     │
                                     ▼
                 ┌────────────────────────────────────────┐
                 │        Retrieval Layer (RAG)           │
                 ├────────────────────────────────────────┤
                 │  FAISS Index per bank:                 │
                 │      storage/sbi/index.faiss           │
                 │      storage/hdfc/index.faiss          │
                 │      storage/common/index.faiss        │
                 │                                        │
                 │  Metadata per chunk (source file, id)  │
                 └────────────────────┬───────────────────┘
                                      │
                                      ▼
                    ┌────────────────────────────────┐
                    │  Gemini LLM (google-genai)     │
                    │  Structured JSON output only   │
                    └────────────────────────────────┘

**🗂️ Folder Structure**

Policy Bot/
│
├── backend/
│   ├── api/
│   │   ├── main.py           → FastAPI server
│   │   ├── chat_logic.py     → Gemini RAG + JSON logic
│   │   ├── session_store.py  → Chat history per session
│   │
│   ├── ingest/
│   │   ├── build_indexes.py  → Creates FAISS indexes
│   │   ├── pdf_utils.py      → PDF reading + chunking
│   │   ├── config.py         → Data folder paths
│
├── data/
│   ├── common/               → Generic PDFs
│   ├── sbi/                  → SBI PDFs
│   ├── hdfc/                 → HDFC PDFs
│   ├── icici/                → ICICI PDFs
│
├── storage/
│   ├── common/               → FAISS index + metadata.pkl
│   ├── sbi/
│   ├── hdfc/
│
├── ui/
│   ├── app.py                → Streamlit floating chat UI
│
├── .env                      → Gemini API key
├── README.md
└── requirements.txt

🔧 Setup Instructions
1️⃣ Clone the project
git clone <your repo url>
cd "Policy Bot"

2️⃣ Create & activate virtual environment
Windows:
python -m venv bankpolicy
bankpolicy\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Add your .env
GEMINI_API_KEY=your-key-here
GEMINI_MODEL=gemini-1.5-flash

5️⃣ Place PDFs in the data/ folders

Example:

data/sbi/*.pdf
data/hdfc/*.pdf
data/common/*.pdf

6️⃣ Build indexes
python -m backend.ingest.build_indexes

7️⃣ Start backend (FastAPI)
uvicorn backend.api.main:app --reload

8️⃣ Run UI
streamlit run ui/app.py


The UI opens at:

http://localhost:8501

✨ Features & Highlights
✔ Retrieval Augmented Generation (RAG)

Only policy chunks relevant to the question are sent to the LLM.

✔ Bank-specific or general mode

If the user selects a bank → search that bank + common
If no bank selected → search all banks

✔ Clean structured JSON

LLM is forced to output only valid JSON.
A custom parser (safe_parse_llm_json) handles messy outputs safely.

✔ Session-based chat memory

Each chat session uses its own conversation history.

✔ Floating search/chat UI

Streamlit UI replicates a modern chat-style experience.

✔ Future-ready

Modular architecture allows easily adding:

More banks

Logging

Auth

Better vector engines (Qdrant, Pinecone)

