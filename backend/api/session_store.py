# backend/api/session_store.py
from typing import Dict, Any, List


class SessionStore:
    """
    Very simple in-memory session store.

    In real production, replace with Redis/DB.
    """

    def __init__(self):
        # session_id -> {"history": [...], "bank": str | None}
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def get_session(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self._sessions:
            self._sessions[session_id] = {"history": [], "bank": None}
        return self._sessions[session_id]

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ):
        session = self.get_session(session_id)
        session["history"].append({"role": role, "content": content})

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        return self.get_session(session_id)["history"]

    def set_bank(self, session_id: str, bank: str | None):
        session = self.get_session(session_id)
        session["bank"] = bank

    def get_bank(self, session_id: str) -> str | None:
        return self.get_session(session_id)["bank"]


session_store = SessionStore()
