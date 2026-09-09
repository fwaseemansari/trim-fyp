"""
Central config module. Loads API keys and shared settings from .env so
every other module (llm_clients, token_analysis, etc.) imports from here
instead of calling os.getenv() directly everywhere.
"""

import os
from dotenv import load_dotenv

# Loads variables from a .env file in the repo root into the process
# environment. Safe to call multiple times / on import.
load_dotenv(override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Default models per backend
GROQ_MODEL = "openai/gpt-oss-20b"
OPENAI_MODEL = "gpt-4o-mini"


def check_keys_loaded() -> None:
    """Quick sanity check — call this once when setting up to confirm
    the .env file is actually being picked up, without printing the
    key values themselves."""
    missing = []
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")

    if missing:
        print(f"[config] Missing keys in .env: {', '.join(missing)}")
    else:
        print("[config] Both API keys loaded successfully.")


if __name__ == "__main__":
    check_keys_loaded()
