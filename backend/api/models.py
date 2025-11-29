# backend/api/models.py
from typing import List, Optional
from pydantic import BaseModel


class QuestionRequest(BaseModel):
    question: str
    bank: Optional[str] = None  # e.g. "HDFC", "SBI", or None
    session_id: str             # to keep chat context alive
    top_k_per_index: int = 5    # optional override


class SourceItem(BaseModel):
    bank: str
    document_name: str
    snippet: str


class AnswerResponse(BaseModel):
    summary: str           # Section 1A
    steps: str             # Section 1B
    sources: List[SourceItem]  # Section 2
    cost_saving_tips: str  # Section 3
