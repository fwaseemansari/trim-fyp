"""Thin pipeline wrappers around one configurable ``Pipeline.process()`` entry point."""

from pipeline.llm_clients import LLMClient
from pipeline.config import PipelineConfig
from token_analysis.counter import TokenAnalyzer
from prompt_compression.compressor import build_compressor
from context_manager.manager import ContextManager
from pipeline.prompts import build_prompt


class Pipeline:
    """Orchestrates compression, context management and the LLM call.

    One entry point, ``process()``, routes by input type:

    * ``context`` (a string)      -> single-turn QA: [compress] -> prompt -> LLM.
    * ``session`` (list of turns) -> multi-turn: context manager and optional
      compressor (order set by ``config.stage_order``) -> prompt -> LLM.

    Each call returns a dict with the response, the token/cost analysis from
    TokenAnalyzer, the final context sent to the model, and, when compression
    ran, the CompressionResult and its separate LLM cost. ``end_to_end_cost_usd``
    adds that compression cost to the answer-call cost, so LLM-based compression
    is not under-counted.

    Components (llm, analyzer, compressor, context_manager) can be injected for
    testing. ``evaluator`` is accepted but not used yet: answer scoring lives in
    evaluation/run_experiment.py, not in the pipeline.
    """

    def __init__(self, llm=None, analyzer=None, compressor=None, context_manager=None, evaluator=None,
                 config: PipelineConfig | None = None):
        self.llm = llm or LLMClient()
        self.analyzer = analyzer or TokenAnalyzer()
        self.config = config or PipelineConfig()
        self.compressor = compressor
        self.context_manager = context_manager
        self.evaluator = evaluator

    def _get_compressor(self):
        """Build the compressor once from the config and reuse it.

        Returns None when compression is disabled.
        """
        if not self.config.compression_enabled:
            return None
        if self.compressor is None:
            self.compressor = build_compressor(
                self.config.compression_method,
                backend=self.config.backend,
                scorer=self.config.compression_scorer,
                llm_client=self.llm,
                window_size=self.config.compression_window_size,
            )
        return self.compressor

    def process(self, *, query: str, context: str = "", session: list | None = None) -> dict:
        """Single entry point; routes by input type rather than a mode flag.

        Args:
            query: the question to answer.
            context: passage for single-turn QA (ignored when ``session`` is given).
            session: list of turn dicts with ``speaker`` and ``text``; selects the
                multi-turn path when not None.

        Returns:
            dict with response, latency_ms, token/cost analysis, the context sent
            to the model, compression_result (or None), compression_cost_usd and
            end_to_end_cost_usd. Multi-turn calls also return per_turn_tokens and
            context_strategy.
        """
        if session is not None:
            return self._process_conversation(session, query)
        return self._process_single(query, context)

    def _process_single(self, query: str, context: str) -> dict:
        """Single-turn path: optionally compress ``context``, build the QA prompt, call the LLM."""
        compression_result = None
        if context and self.config.compression_enabled:
            compression_result = self._get_compressor().compress(
                context, query=query, level=self.config.compression_level
            )
            context = compression_result.text

        full_prompt = self._build_qa_prompt(query, context)
        result = self.llm.generate(full_prompt, backend=self.config.backend)
        model = self.llm.model_name(self.config.backend)
        analysis = self.analyzer.log_run(
            prompt=full_prompt,
            response=result.response,
            backend=self.config.backend,
            model=model,
            latency_ms=result.latency_ms,
        )
        compression_cost = 0.0
        if compression_result is not None:
            from token_analysis.pricing import estimate_cost
            compression_cost = estimate_cost(
                model, compression_result.compression_input_tokens,
                compression_result.compression_output_tokens,
            )
        return {
            "query": query,
            "response": result.response,
            "latency_ms": result.latency_ms,
            "compressed_context": context,
            "compression_result": compression_result,
            "compression_cost_usd": compression_cost,
            "end_to_end_cost_usd": analysis["cost_usd"] + compression_cost,
            **analysis,
        }

    def _process_conversation(self, session: list, query: str) -> dict:
        """Multi-turn path.

        stage_order "manager_then_compressor" (default): the context manager
        selects turns, then the compressor shrinks that assembled text.
        stage_order "compressor_then_manager" (experimental): the whole session
        is compressed as one text and fed to the manager as a single turn.
        Raises ValueError for any other stage_order.
        """
        cm = self.context_manager or ContextManager(
            strategy=self.config.context_strategy,
            max_turns=self.config.max_turns,
            max_tokens=self.config.max_tokens,
            llm_client=self.llm if self.config.context_strategy == "memory_summarization" else None,
        )
        per_turn_tokens = []

        if self.config.stage_order == "manager_then_compressor":
            for turn in session:
                cm.add_turn(turn, query=query)
                per_turn_tokens.append(cm.total_tokens())
            context = cm.get_context()
            compression_result = None
            if self.config.compression_enabled:
                compression_result = self._get_compressor().compress(
                    context, query=query, level=self.config.compression_level
                )
                context = compression_result.text
        elif self.config.stage_order == "compressor_then_manager":
            raw_context = " ".join(f"{t.get('speaker', 'unknown')}: {t.get('text', '')}" for t in session)
            if self.config.compression_enabled:
                compression_result = self._get_compressor().compress(
                    raw_context, query=query, level=self.config.compression_level
                )
                compressed_context = compression_result.text
            else:
                compression_result = None
                compressed_context = raw_context
            # Experimental order: feed the compressed turns as one bounded turn.
            cm.add_turn({"speaker": "compressed_context", "text": compressed_context}, query=query)
            per_turn_tokens.append(cm.total_tokens())
            context = cm.get_context()
        else:
            raise ValueError("stage_order must be 'manager_then_compressor' or 'compressor_then_manager'")

        full_prompt = build_prompt(task_type="conversation", query=query, context=context)
        result = self.llm.generate(full_prompt, backend=self.config.backend)
        model = self.llm.model_name(self.config.backend)
        analysis = self.analyzer.log_run(
            prompt=full_prompt,
            response=result.response,
            backend=self.config.backend,
            model=model,
            latency_ms=result.latency_ms,
        )
        compression_cost = 0.0
        if compression_result is not None:
            from token_analysis.pricing import estimate_cost
            compression_cost = estimate_cost(
                model, compression_result.compression_input_tokens,
                compression_result.compression_output_tokens,
            )
        return {
            "query": query,
            "response": result.response,
            "latency_ms": result.latency_ms,
            "per_turn_tokens": per_turn_tokens,
            "context_strategy": self.config.context_strategy,
            "context": context,
            "compression_result": compression_result,
            "compression_cost_usd": compression_cost,
            "end_to_end_cost_usd": analysis["cost_usd"] + compression_cost,
            **analysis,
        }

    def _build_qa_prompt(self, query: str, context: str, conversation: bool = False) -> str:
        return build_prompt(
            task_type="conversation" if conversation else self.config.task_type,
            query=query,
            context=context,
        )


def run(query: str, context: str = "", compression_enabled: bool = False,
        compression_method: str = "extractive", compression_level: float = 0.5,
        backend: str = "groq", llm_client=None, analyzer=None, compression_scorer: str = "tfidf",
        compression_window_size: int = 2) -> dict:
    """Backward-compatible Week 1/2 wrapper around Pipeline.process().

    Note its defaults (method "extractive", scorer "tfidf") differ from
    PipelineConfig's ("selective", "bi_encoder"); pass them explicitly in
    experiments.
    """
    cfg = PipelineConfig(
        compression_enabled=compression_enabled,
        compression_method=compression_method,
        compression_scorer=compression_scorer,
        compression_level=compression_level,
        compression_window_size=compression_window_size,
        backend=backend,
    )
    return Pipeline(llm=llm_client, analyzer=analyzer, config=cfg).process(
        query=query, context=context
    )


def run_conversation(session: list, query: str, context_strategy: str = "sliding_window",
                     max_turns: int = 6, max_tokens: int = 1000, backend: str = "groq",
                     llm_client=None, analyzer=None, compression_enabled: bool = False,
                     compression_method: str = "selective", compression_scorer: str = "bi_encoder",
                     compression_level: float = 0.5, compression_window_size: int = 2) -> dict:
    """Backward-compatible multi-turn wrapper around Pipeline.process(session=...).

    Compression is off by default; the context strategy defaults to
    "sliding_window".
    """
    cfg = PipelineConfig(
        compression_enabled=compression_enabled,
        compression_method=compression_method,
        compression_scorer=compression_scorer,
        compression_level=compression_level,
        compression_window_size=compression_window_size,
        backend=backend,
        context_strategy=context_strategy,
        max_turns=max_turns,
        max_tokens=max_tokens,
    )
    return Pipeline(llm=llm_client, analyzer=analyzer, config=cfg).process(
        query=query, session=session
    )