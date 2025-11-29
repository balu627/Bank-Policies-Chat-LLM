# backend/api/chat_logic.py
import json
from typing import List, Dict, Any, Optional

from google import genai

from config.settings import GEMINI_API_KEY, GEMINI_MODEL
from backend.api.session_store import session_store

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in environment or .env")

# Create Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)


def build_context_block(retrieved_docs: List[Dict[str, Any]]) -> str:
    """
    Build a text block summarizing all relevant policy chunks.

    Format:
    [BANK: hdfc] [DOCUMENT: xyz.pdf]
    <merged text>

    ...
    """
    parts: List[str] = []
    for doc in retrieved_docs:
        bank = doc["bank"]
        document_name = doc["document_name"]
        merged_text = doc["merged_text"]

        part = (
            f"[BANK: {bank}] [DOCUMENT: {document_name}]\n"
            f"{merged_text}\n"
            "----\n"
        )
        parts.append(part)
    return "\n".join(parts)


def build_sources_for_llm(retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Compact representation of sources: one per bank+document, with a short snippet.
    """
    sources_map: Dict[tuple, Dict[str, str]] = {}

    for doc in retrieved_docs:
        key = (doc["bank"], doc["document_name"])
        if key not in sources_map:
            text = doc["merged_text"]
            snippet_lines = text.strip().splitlines()
            snippet = " ".join(snippet_lines)[:400]  # limit length
            sources_map[key] = {
                "bank": doc["bank"],
                "document_name": doc["document_name"],
                "snippet": snippet,
            }

    return list(sources_map.values())


def _history_to_text(history: List[Dict[str, str]]) -> str:
    """
    Convert our stored chat history to a simple text block.
    """
    lines = []
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines)


def safe_parse_llm_json(raw: str) -> Dict[str, Any]:
    """
    Try to robustly parse JSON coming from the LLM.

    - Strips markdown fences like ```json ... ```
    - Tries to extract the first {...} block.
    - Falls back to a minimal dict if parsing fails.
    """
    text = raw.strip()

    # 1) Remove markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop first line (``` or ```json)
        lines = lines[1:]
        # Drop last line if it's ``` 
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # 2) Try direct json.loads
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 3) Try to extract the first {...} span
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # 4) Final fallback: wrap raw text into minimal structure
    return {
        "summary": raw,
        "steps": "",
        "sources": [],
        "cost_saving_tips": "",
    }


def generate_answer(
    question: str,
    bank: Optional[str],
    session_id: str,
    retrieved_docs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Call the LLM (Gemini) and return structured answer:
    {
      "summary": str,
      "steps": str,
      "sources": [...],
      "cost_saving_tips": str
    }
    """
    # Chat history for continuity
    history = session_store.get_history(session_id)

    policy_context = build_context_block(retrieved_docs)
    sources_for_llm = build_sources_for_llm(retrieved_docs)
    history_text = _history_to_text(history)

    # System instructions: describe 3 sections and constraints.
    # IMPORTANT: We tell Gemini to respond with JSON only.
    system_instructions = """
You are an assistant that answers user questions strictly based on
given bank policy documents and general banking knowledge.

You MUST always respond as a JSON object with keys:
  "summary": string            (Section 1A, policy-based)
  "steps": string              (Section 1B, policy-based step-wise process)
  "sources": array of objects  (Section 2)
      Each object: { "bank": string, "document_name": string, "snippet": string }
  "cost_saving_tips": string   (Section 3, general/online info allowed)

Rules:
- "summary" and "steps" MUST use ONLY the provided policy_context text.
  Do NOT invent new rules or numbers.
- If the context is incomplete or does not specify something, clearly say that.
- For "steps", if the policy describes a process (e.g., account opening, loan application,
  credit card application), give clear numbered steps based ONLY on the document content.
- If the policy does NOT specify a clear process, say that and only explain what it does say.
- "sources" must reflect which bank and which document were used, with short snippets
  that are cleaned up but keep the same meaning as the original policy text.
- "cost_saving_tips" may use general banking and online knowledge, but MUST clearly say
  that this section is based on general/online information and not directly from the policy documents.
- VERY IMPORTANT: Output MUST be a single JSON object only.
  Do NOT write any text before or after the JSON.
  Do NOT wrap the JSON in ``` or any other formatting.
"""

    # Bundle everything into a single prompt string for Gemini
    user_payload = {
        "question": question,
        "bank_context": bank or "",
        "policy_context": policy_context,
        "structured_sources_hint": sources_for_llm,
        "chat_history": history_text,
    }

    prompt = (
        system_instructions
        + "\n\nHere is the data for this request as a JSON object:\n"
        + json.dumps(user_payload, ensure_ascii=False)
        + "\n\nNow produce the answer as a JSON object with the required keys."
    )

    # --- Gemini call ---
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    # For Gemini, the main text is in response.text
    content = response.text or ""
    # print("Gemini raw response:", repr(content))

    # Robust JSON parsing
    parsed = safe_parse_llm_json(content)

    # Ensure sources are in correct shape
    sources = parsed.get("sources", [])
    if not isinstance(sources, list):
        sources = []

    return {
        "summary": parsed.get("summary", "").strip(),
        "steps": parsed.get("steps", "").strip(),
        "sources": sources,
        "cost_saving_tips": parsed.get("cost_saving_tips", "").strip(),
    }
