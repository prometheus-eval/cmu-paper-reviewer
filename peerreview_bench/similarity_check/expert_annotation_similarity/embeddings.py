"""
Pluggable embedding backends for similarity_check.

Backends:
    - `qwen3`           : local sentence-transformers model Qwen3-Embedding-8B
    - `azure`           : `azure_ai/text-embedding-3-large` via LiteLLM proxy
    - `gemini`          : `gemini-embedding-001` via LiteLLM proxy

There is NO TF-IDF fallback. If the required library is missing or the API
is unreachable, we fail loudly rather than silently degrade.

All backends expose a common interface:

    emb = get_backend('qwen3')
    sim_matrix = emb.pairwise(texts_a, texts_b)        # [len(a), len(b)] cosine sims
    vec = emb.embed(['hello'])                          # [1, dim] numpy array
"""

from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

# Import the shared LiteLLM client from metareview_bench/ — same singleton the
# metareview_bench module uses. We add its path to sys.path so we can import the
# small wrapper without having to duplicate the code.
# Path: expert_annotation_similarity/embeddings.py → ../../../metareview_bench
_META_REVIEW_DIR = Path(__file__).resolve().parent.parent.parent / "metareview_bench"
if str(_META_REVIEW_DIR) not in sys.path:
    sys.path.insert(0, str(_META_REVIEW_DIR))


def _cosine_matrix(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between every row of emb_a and every row of emb_b."""
    if emb_a.size == 0 or emb_b.size == 0:
        return np.zeros((emb_a.shape[0], emb_b.shape[0]), dtype=np.float32)
    na = np.linalg.norm(emb_a, axis=1, keepdims=True)
    nb = np.linalg.norm(emb_b, axis=1, keepdims=True)
    na[na == 0] = 1.0
    nb[nb == 0] = 1.0
    return ((emb_a / na) @ (emb_b / nb).T).astype(np.float32)


class EmbeddingBackend(ABC):
    name: str = "unknown"

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Return an [N, D] numpy array of L2-normalized embeddings."""

    def pairwise(self, texts_a: Iterable[str], texts_b: Iterable[str]) -> np.ndarray:
        a = [t or "" for t in texts_a]
        b = [t or "" for t in texts_b]
        if not a or not b:
            return np.zeros((len(a), len(b)), dtype=np.float32)
        emb_a = self.embed(a)
        emb_b = self.embed(b)
        return _cosine_matrix(emb_a, emb_b)


# ----------------------------------------------------------------------
# Backend 1: Qwen3-Embedding-8B via sentence-transformers (local)
# ----------------------------------------------------------------------

class QwenEmbedding(EmbeddingBackend):
    name = "qwen3"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-8B",
        device: Optional[str] = None,
        batch_size: int = 16,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "The `sentence-transformers` package is required for the Qwen "
                "backend. Install with `pip install sentence-transformers`."
            ) from e

        self._st = SentenceTransformer(model_name, device=device)
        self._batch_size = batch_size
        self.name = f"qwen3:{model_name}"

    def embed(self, texts: List[str]) -> np.ndarray:
        vecs = self._st.encode(
            texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)


# ----------------------------------------------------------------------
# Backend 2: azure_ai/text-embedding-3-large via LiteLLM proxy
# ----------------------------------------------------------------------

class LiteLLMEmbedding(EmbeddingBackend):
    """Any embedding model routed via the CMU LiteLLM proxy.

    Includes retry-with-exponential-backoff on 429 / RateLimitError so the
    embedding backend survives transient TPM/RPM throttling from the
    upstream provider (notably Gemini's short-window quota on the CMU
    LiteLLM proxy). A small inter-batch pause reduces burstiness.
    """

    # Backoffs in seconds for consecutive rate-limit retries. Up to ~8 min
    # total wait before giving up — long enough for most per-minute quotas
    # to reset.
    _RATE_LIMIT_BACKOFFS = [10, 30, 60, 120, 240]

    def __init__(
        self,
        model: str,
        *,
        batch_size: int = 64,
        dimensions: Optional[int] = None,
        inter_batch_sleep: float = 0.5,
    ):
        from litellm_client import call_embedding  # noqa: F401 — just validating import
        self._model = model
        self._batch_size = batch_size
        self._dimensions = dimensions
        self._inter_batch_sleep = inter_batch_sleep
        bare = model[len("litellm_proxy/"):] if model.startswith("litellm_proxy/") else model
        self.name = f"litellm:{bare.replace('/', '_')}"

    def _call_with_retry(self, batch: List[str]) -> List[List[float]]:
        """Call the proxy with exponential backoff on rate-limit errors."""
        import time
        from litellm_client import call_embedding

        last_err: Optional[Exception] = None
        for attempt, wait in enumerate([0] + list(self._RATE_LIMIT_BACKOFFS)):
            if wait:
                print(f'    rate-limited by {self.name}; sleeping {wait}s '
                      f'before retry {attempt}/{len(self._RATE_LIMIT_BACKOFFS)}...',
                      flush=True)
                time.sleep(wait)
            try:
                return call_embedding(
                    model=self._model,
                    texts=batch,
                    dimensions=self._dimensions,
                )
            except Exception as e:
                last_err = e
                # Retry only on rate-limit-ish errors. Everything else
                # re-raises immediately so we don't mask real bugs.
                msg = str(e).lower()
                is_rate_limit = (
                    '429' in msg
                    or 'rate' in msg and 'limit' in msg
                    or 'resource_exhausted' in msg
                    or 'resource exhausted' in msg
                    or type(e).__name__ == 'RateLimitError'
                )
                if not is_rate_limit:
                    raise
                if attempt == len(self._RATE_LIMIT_BACKOFFS):
                    # Final attempt exhausted — propagate.
                    raise
        # Unreachable but keep type-checker happy.
        assert last_err is not None
        raise last_err

    def embed(self, texts: List[str]) -> np.ndarray:
        import time
        from tqdm import tqdm

        vectors: List[List[float]] = []
        n_batches = (len(texts) + self._batch_size - 1) // self._batch_size
        batch_indices = list(range(0, len(texts), self._batch_size))
        pbar = tqdm(
            total=len(texts),
            desc=self.name,
            unit='text',
            dynamic_ncols=True,
            mininterval=0.5,
        )
        try:
            for bi, i in enumerate(batch_indices):
                batch = texts[i : i + self._batch_size]
                batch = [(t or " ") for t in batch]  # empty strings confuse some providers
                vecs = self._call_with_retry(batch)
                vectors.extend(vecs)
                pbar.update(len(batch))
                if self._inter_batch_sleep and bi < n_batches - 1:
                    time.sleep(self._inter_batch_sleep)
        finally:
            pbar.close()

        arr = np.asarray(vectors, dtype=np.float32)
        # L2-normalize so cosine sim = dot product
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


class AzureTextEmbedding3Large(LiteLLMEmbedding):
    """Convenience wrapper for azure_ai/text-embedding-3-large."""

    def __init__(self, **kwargs):
        super().__init__(model="litellm_proxy/azure_ai/text-embedding-3-large", **kwargs)
        self.name = "azure_text_embedding_3_large"


class GeminiEmbedding001(LiteLLMEmbedding):
    """Convenience wrapper for gemini-embedding-001."""

    def __init__(self, **kwargs):
        super().__init__(model="litellm_proxy/gemini/gemini-embedding-001", **kwargs)
        self.name = "gemini_embedding_001"


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------

BACKEND_REGISTRY = {
    "qwen3": QwenEmbedding,
    "azure": AzureTextEmbedding3Large,
    "azure_text_embedding_3_large": AzureTextEmbedding3Large,
    "gemini": GeminiEmbedding001,
    "gemini_embedding_001": GeminiEmbedding001,
}


def get_backend(name_or_model: str, **kwargs) -> EmbeddingBackend:
    """Look up a backend by short name, or treat the arg as a LiteLLM
    embedding model id (with or without the `litellm_proxy/` prefix)."""
    key = name_or_model.lower()
    if key in BACKEND_REGISTRY:
        return BACKEND_REGISTRY[key](**kwargs)
    # Fall through: treat as a generic litellm embedding model
    return LiteLLMEmbedding(model=name_or_model, **kwargs)
