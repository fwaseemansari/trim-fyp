"""Guards for the cleanup: no dead selector, one model map, run() defaults from the config."""

import importlib.util
import inspect
import re
from pathlib import Path

from pipeline import config, pipeline as pipe
from pipeline.config import PipelineConfig
from prompt_compression.compressor import build_compressor

ROOT = Path(__file__).resolve().parent.parent
SKIP = {"venv", ".venv", ".git", "__pycache__", "node_modules"}


def _python_files():
    return [p for p in ROOT.rglob("*.py")
            if not SKIP & set(p.relative_to(ROOT).parts) and "tests" not in p.relative_to(ROOT).parts]


def test_budget_selector_is_gone():
    assert importlib.util.find_spec("prompt_compression.selectors") is None
    assert not any("BudgetSelector" in p.read_text(encoding="utf-8") for p in _python_files())


def test_there_is_a_single_backend_to_model_map():
    offenders = [str(p.relative_to(ROOT)) for p in _python_files()
                 if re.search(r"^\s*BACKEND_MODEL\s*=", p.read_text(encoding="utf-8"), re.M)]
    assert offenders == []


def test_compressor_model_comes_from_model_for_backend():
    for backend in ("groq", "openai"):
        c = build_compressor("selective", scorer="tfidf", backend=backend)
        assert c.model == config.model_for_backend(backend)


def test_run_defaults_match_pipeline_config():
    params = inspect.signature(pipe.run).parameters
    cfg = PipelineConfig()
    for name in ("compression_method", "compression_level", "compression_scorer",
                 "compression_window_size", "compression_splitter", "backend"):
        assert params[name].default == getattr(cfg, name), name


def test_pipeline_has_no_unused_evaluator_parameter():
    assert "evaluator" not in inspect.signature(pipe.Pipeline.__init__).parameters