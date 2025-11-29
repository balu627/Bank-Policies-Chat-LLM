# ui/app.py
import uuid
import requests
import streamlit as st

API_BASE_URL = "http://127.0.0.1:8000"


def get_session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id


def init_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of {"role": "user"/"assistant", "content": {...}}


def main():
    st.set_page_config(page_title="Bank Policy Assistant", layout="wide")

    st.title("🏦 Bank Policy Assistant")

    init_chat_history()
    session_id = get_session_id()

    # Fetch available banks from backend
    try:
        resp = requests.get(f"{API_BASE_URL}/")
        resp.raise_for_status()
        data = resp.json()
        available_banks = data.get("available_banks", [])
    except Exception:
        available_banks = []

    # Bank selection (optional)
    with st.sidebar:
        st.subheader("Bank selection")
        bank_options = ["Auto-detect / None"] + available_banks
        selected_bank_label = st.selectbox("Bank", bank_options, index=0)
        selected_bank = (
            None
            if selected_bank_label == "Auto-detect / None"
            else selected_bank_label
        )

    st.markdown("---")

    # Display chat history
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        else:
            # Assistant: has structured sections
            with st.chat_message("assistant"):
                st.markdown("### 1. Policy Summary (from documents only)")
                st.write(content["summary"])

                st.markdown("### 1B. Step-wise Process (from documents only)")
                if content["steps"]:
                    st.write(content["steps"])
                else:
                    st.write("_No specific step-wise process described in the documents._")

                st.markdown("### 2. Sources (policy documents)")
                if content["sources"]:
                    for src in content["sources"]:
                        st.markdown(
                            f"- **Bank:** {src['bank']}  \n"
                            f"  **Document:** {src['document_name']}  \n"
                            f"  **Snippet:** {src['snippet']}"
                        )
                else:
                    st.write("_No clear source documents identified._")

                st.markdown("### 3. Cost-saving tips (general/online info)")
                st.write(content["cost_saving_tips"])

    # Bottom-floating input
    prompt = st.chat_input("Ask about account opening, credit cards, loans, etc.")
    if prompt:
        # Add user message to history
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        # Call backend
        payload = {
            "question": prompt,
            "bank": selected_bank,
            "session_id": session_id,
        }

        try:
            resp = requests.post(f"{API_BASE_URL}/ask", json=payload, timeout=60)
            resp.raise_for_status()
            ans = resp.json()
        except Exception as e:
            # Show error as assistant message
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": {
                        "summary": f"Error calling backend: {e}",
                        "steps": "",
                        "sources": [],
                        "cost_saving_tips": "",
                    },
                }
            )
        else:
            # Add structured answer to history
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": ans,
                }
            )

        st.rerun()


if __name__ == "__main__":
    main()
