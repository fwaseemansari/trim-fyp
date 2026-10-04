"""Relevance scorers shared by selective compression variants."""

from abc import ABC, abstractmethod
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_MODEL_CACHE: dict = {}          # <- module level, outside every class


def _load_cached(kind: str, model_name: str):
    """Load a sentence-transformers model once per process and share it.

    Without this, every new scorer object reloaded the model; over a 300-sample
    run that exhausted Windows virtual memory (OSError 1455).
    kind: "bi" for SentenceTransformer, anything else for CrossEncoder.
    """
    key = (kind, model_name)
    if key not in _MODEL_CACHE:
        if kind == "bi":
            from sentence_transformers import SentenceTransformer
            _MODEL_CACHE[key] = SentenceTransformer(model_name)
        else:
            from sentence_transformers import CrossEncoder
            _MODEL_CACHE[key] = CrossEncoder(model_name)
    return _MODEL_CACHE[key]

class Scorer(ABC):
    """Scores text segments against a query (higher = more relevant).

    Only the ranking matters to the selector. Scores from different scorers are
    NOT comparable: cosine similarity lies in [-1, 1], cross-encoder outputs
    are unbounded logits.
    """

    name = "base"

    @abstractmethod
    def score(self, segments, query: str) -> list[float]:
        """Return one score per segment, in the same order as ``segments``."""
        raise NotImplementedError


class TfidfScorer(Scorer):
    """Cosine similarity of TF-IDF vectors (lexical overlap; no stemming)."""

    name = "tfidf"

    def score(self, segments, query: str) -> list[float]:
        if not segments:
            return []
        texts = [s.text for s in segments]
        vectorizer = TfidfVectorizer().fit(texts + [query or ""])
        matrix = vectorizer.transform(texts)
        q = vectorizer.transform([query or ""])
        return cosine_similarity(matrix, q).flatten().astype(float).tolist()


class BiEncoderScorer(Scorer):
    """Semantic scorer using a sentence-transformers bi-encoder."""

    name = "bi_encoder"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", model=None):
        self.model_name = model_name
        self._model = model

    @property
    def model(self):
        if self._model is None:
            self._model = _load_cached("bi", self.model_name)
        return self._model

    def score(self, segments, query: str) -> list[float]:
        if not segments:
            return []
        embeddings = self.model.encode(
            [query or ""] + [s.text for s in segments],
            normalize_embeddings=True,
        )
        q = embeddings[0]
        return np.dot(embeddings[1:], q).astype(float).tolist()


class CrossEncoderScorer(Scorer):
    """Pairwise relevance scorer. Loaded lazily so Week 3 does not pay its cost."""

    name = "cross_encoder"

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", model=None):
        self.model_name = model_name
        self._model = model

    @property
    def model(self):
        if self._model is None:
            self._model = _load_cached("ce", self.model_name)
        return self._model

    def score(self, segments, query: str) -> list[float]:
        if not segments:
            return []
        pairs = [(query or "", s.text) for s in segments]
        scores = self.model.predict(pairs)
        return np.asarray(scores, dtype=float).tolist()