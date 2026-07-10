import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from classroom_pipeline.verifier.model import action_match_score


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _tokenize(text: str) -> List[str]:
    text = _normalize_text(text)
    if not text:
        return []
    tokens = text.replace("_", " ").replace("-", " ").split()
    if len(text) >= 3:
        grams = [text[i : i + 3] for i in range(len(text) - 2)]
    else:
        grams = [text] if text else []
    return tokens + grams


def _dot(a: List[float], b: List[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def _norm(a: List[float]) -> float:
    return float(sum(x * x for x in a) ** 0.5)


def _cosine(a: List[float], b: List[float]) -> float:
    denom = _norm(a) * _norm(b)
    if denom <= 1e-8:
        return 0.0
    return float(_dot(a, b) / denom)


class TextSimilarityScorer:
    def __init__(
        self,
        *,
        mode: str = "rule",
        cache_path: str = "output/cache/text_embeddings.pkl",
        embedding_backend: str = "auto",
        embedding_model: str = "",
        embedding_dim: int = 128,
        hybrid_alpha: float = 0.5,
    ) -> None:
        self.mode = str(mode).strip().lower() if mode else "rule"
        if self.mode not in {"rule", "embedding", "hybrid"}:
            self.mode = "rule"
        self.embedding_dim = max(16, int(embedding_dim))
        self.hybrid_alpha = _clamp01(float(hybrid_alpha))
        self.cache_path = Path(cache_path).resolve()
        self.embedding_model = str(embedding_model or "").strip()
        self.embedding_backend = str(embedding_backend or "auto").strip().lower()
        self.cache: Dict[str, List[float]] = {}
        self.dirty = False

        self._sentence_model = None
        self.backend_used = "rule_only"
        self.backend_reason = "rule_mode"

        self._load_cache()
        if self.mode in {"embedding", "hybrid"}:
            self._init_embedding_backend()

    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            self.cache = {}
            return
        try:
            with self.cache_path.open("rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                normalized: Dict[str, List[float]] = {}
                for key, value in data.items():
                    if isinstance(key, str) and isinstance(value, list):
                        normalized[key] = [float(x) for x in value]
                self.cache = normalized
            else:
                self.cache = {}
        except Exception:
            self.cache = {}

    def _init_embedding_backend(self) -> None:
        prefer_sentence = self.embedding_backend in {"auto", "sentence_transformers"}
        if prefer_sentence:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                model_name = self.embedding_model or "all-MiniLM-L6-v2"
                self._sentence_model = SentenceTransformer(model_name, local_files_only=True)
                self.backend_used = "sentence_transformers"
                self.backend_reason = "loaded_local_model"
                return
            except Exception:
                self._sentence_model = None
        self.backend_used = "hashing_fallback"
        self.backend_reason = "no_local_embedding_model"

    def _cache_key(self, text: str) -> str:
        prefix = f"{self.backend_used}|{self.embedding_model or 'default'}|{self.embedding_dim}|"
        return prefix + _normalize_text(text)

    def _hash_embedding(self, text: str) -> List[float]:
        vec = [0.0] * self.embedding_dim
        tokens = _tokenize(text)
        if not tokens:
            return vec
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            idx = int(digest[:8], 16) % self.embedding_dim
            sign = 1.0 if (int(digest[8:10], 16) % 2 == 0) else -1.0
            vec[idx] += sign
        n = _norm(vec)
        if n > 1e-8:
            vec = [v / n for v in vec]
        return vec

    def _embed(self, text: str) -> List[float]:
        text = _normalize_text(text)
        if not text:
            return [0.0] * self.embedding_dim
        key = self._cache_key(text)
        if key in self.cache:
            return self.cache[key]

        if self._sentence_model is not None:
            try:
                arr = self._sentence_model.encode([text], normalize_embeddings=True)
                vec = [float(x) for x in arr[0].tolist()]
            except Exception:
                vec = self._hash_embedding(text)
                self.backend_used = "hashing_fallback"
                self.backend_reason = "sentence_transformers_runtime_error"
        else:
            vec = self._hash_embedding(text)

        self.cache[key] = vec
        self.dirty = True
        return vec

    def score(self, *, event_type: str, query_text: str, action_label: str) -> Tuple[float, Dict[str, str]]:
        rule_score = _clamp01(action_match_score(event_type, query_text, action_label))
        if self.mode == "rule":
            return rule_score, {
                "mode": "rule",
                "backend": "rule_only",
                "backend_reason": "rule_mode",
            }

        left_text = f"{event_type} {query_text}".strip()
        right_text = str(action_label or "")
        emb_score = _clamp01((1.0 + _cosine(self._embed(left_text), self._embed(right_text))) * 0.5)

        if self.mode == "embedding":
            return emb_score, {
                "mode": "embedding",
                "backend": self.backend_used,
                "backend_reason": self.backend_reason,
            }

        score = _clamp01(self.hybrid_alpha * rule_score + (1.0 - self.hybrid_alpha) * emb_score)
        return score, {
            "mode": "hybrid",
            "backend": self.backend_used,
            "backend_reason": self.backend_reason,
        }

    def close(self) -> None:
        if not self.dirty:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("wb") as f:
            pickle.dump(self.cache, f)
        self.dirty = False

