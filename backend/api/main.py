# backend/api/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.api.models import QuestionRequest, AnswerResponse, SourceItem
from backend.api.session_store import session_store
from backend.api.retrieval import retrieval_engine
from backend.api.chat_logic import generate_answer


app = FastAPI(
    title="Bank Policy Assistant",
    description="Ask questions based on bank policy PDFs.",
    version="1.0.0",
)

# CORS (so Streamlit frontend can call backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in real deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "Bank Policy Assistant API is running.",
        "available_banks": retrieval_engine.available_banks,
    }


@app.post("/ask", response_model=AnswerResponse)
def ask_question(req: QuestionRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Use session store for bank context if not explicitly provided
    session_bank = session_store.get_bank(req.session_id)

    bank = req.bank or session_bank

    # Basic bank detection from question (optional, simple)
    if not bank:
        lower_q = req.question.lower()
        for b in retrieval_engine.available_banks:
            if b.lower() in lower_q:
                bank = b
                break

    # Store/update bank in session
    if bank:
        session_store.set_bank(req.session_id, bank)

    # Retrieval
    try:
        retrieved_docs = retrieval_engine.retrieve(
            question=req.question,
            bank=bank,
            top_k_per_index=req.top_k_per_index,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Generate answer with LLM
    answer_data = generate_answer(
        question=req.question,
        bank=bank,
        session_id=req.session_id,
        retrieved_docs=retrieved_docs,
    )

    # Save to session history (so future Qs have context)
    session_store.add_message(
        req.session_id,
        role="user",
        content=req.question,
    )
    # We store a compact summary of the assistant answer
    session_store.add_message(
        req.session_id,
        role="assistant",
        content=answer_data["summary"],
    )

    # Map sources into Pydantic model
    sources_items = [
        SourceItem(
            bank=s.get("bank", ""),
            document_name=s.get("document_name", ""),
            snippet=s.get("snippet", ""),
        )
        for s in answer_data["sources"]
    ]

    return AnswerResponse(
        summary=answer_data["summary"],
        steps=answer_data["steps"],
        sources=sources_items,
        cost_saving_tips=answer_data["cost_saving_tips"],
    )
