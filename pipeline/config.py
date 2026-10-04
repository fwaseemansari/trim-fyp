"""Central configuration for API keys, models and pipeline defaults."""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv(override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_MODEL = "openai/gpt-oss-20b"
OPENAI_MODEL = "gpt-4o-mini"


@dataclass
class PipelineConfig:
    """Settings for one Pipeline.

    compression_*: whether and how to compress (method, scorer, level = token
        budget fraction, window size in sentences).
    backend: "groq" or "openai".
    context_strategy / max_turns / max_tokens: multi-turn context manager settings.
    stage_order: "manager_then_compressor" or "compressor_then_manager".
    task_type: prompt template key for single-turn input (see prompts.py).
    """

    compression_enabled: bool = False
    compression_method: str = "selective"
    compression_scorer: str = "bi_encoder"
    compression_level: float = 0.5
    compression_window_size: int = 2
    backend: str = "groq"
    context_strategy: str = "sliding_window"
    max_turns: int = 6
    max_tokens: int = 1000
    stage_order: str = "manager_then_compressor"
    task_type: str = "qa"


def model_for_backend(backend: str) -> str:
    """Model name for a backend; raises ValueError for an unknown backend."""
    if backend == "groq":
        return GROQ_MODEL
    if backend == "openai":
        return OPENAI_MODEL
    raise ValueError(f"Unknown backend: {backend}")


def check_keys_loaded() -> None:
    missing = []
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if missing:
        print(f"[config] Missing keys in .env: {', '.join(missing)}")
    else:
        print("[config] Both API keys loaded successfully.")