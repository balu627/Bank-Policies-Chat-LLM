# config/settings.py
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"
LOGS_DIR = BASE_DIR / "logs"

# Embeddings model
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# LLM (OpenAI) settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")  # adjust if needed

# LLM (Gemini) settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-gemini-api-key")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # adjust if needed

# Chunking config
CHUNK_SIZE = 800       # characters
CHUNK_OVERLAP = 200    # characters

# Retrieval config
TOP_K_PER_INDEX = 5
MAX_BANKS_WHEN_NO_BANK_SPECIFIED = 5
