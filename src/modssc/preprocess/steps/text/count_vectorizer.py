from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any

import numpy as np

from modssc.preprocess.errors import PreprocessValidationError
from modssc.preprocess.optional import require
from modssc.preprocess.store import ArtifactStore


class _VisibleTextParser(HTMLParser):
    """Extract human-visible text without executing or repairing markup."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._suppressed_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag.lower() in {"script", "style"}:
            self._suppressed_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style"} and self._suppressed_depth:
            self._suppressed_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._suppressed_depth:
            self.parts.append(data)


def _strip_mime_header(text: str) -> str:
    normalized = text.replace("\r\n", "\n")
    header, separator, body = normalized.partition("\n\n")
    if not separator:
        return normalized
    header_lines = [line.strip() for line in header.splitlines() if line.strip()]
    if header_lines and any(
        line.lower().startswith(("content-type:", "mime-version:", "content-transfer-encoding:"))
        for line in header_lines
    ):
        return body
    return normalized


def _visible_text(text: str) -> str:
    parser = _VisibleTextParser()
    parser.feed(_strip_mime_header(text))
    parser.close()
    return " ".join(part for part in parser.parts if part.strip())


def _texts(raw: Any, *, strip_html: bool = False) -> list[str]:
    if isinstance(raw, np.ndarray):
        values = raw.reshape(-1).tolist()
    elif isinstance(raw, (list, tuple)):
        values = list(raw)
    else:
        values = list(raw)
    texts = [str(value) for value in values]
    if strip_html:
        return [_visible_text(value) for value in texts]
    return texts


@dataclass
class CountVectorizerStep:
    """Deterministic bag-of-words counts for historical text protocols."""

    max_features: int | None = None
    ngram_range: tuple[int, int] = (1, 1)
    min_df: int | float = 1
    max_df: int | float = 1.0
    lowercase: bool = True
    binary: bool = False
    dense: bool = False
    strip_html: bool = False

    _vec: Any = field(default=None, init=False, repr=False)

    def fit(
        self, store: ArtifactStore, *, fit_indices: np.ndarray, rng: np.random.Generator
    ) -> None:
        feature_extraction = require(
            module="sklearn.feature_extraction.text",
            extra="preprocess-sklearn",
            purpose="bag-of-words count vectorization",
        )
        texts = _texts(store.require("raw.X"), strip_html=bool(self.strip_html))
        idx = np.asarray(fit_indices, dtype=np.int64)
        if idx.ndim != 1:
            raise PreprocessValidationError("fit_indices must be 1D")
        if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= len(texts)):
            raise PreprocessValidationError("fit_indices are outside raw.X")
        vectorizer = feature_extraction.CountVectorizer(
            max_features=self.max_features,
            ngram_range=tuple(self.ngram_range),
            min_df=self.min_df,
            max_df=self.max_df,
            lowercase=bool(self.lowercase),
            binary=bool(self.binary),
            dtype=np.float32,
        )
        self._vec = vectorizer.fit([texts[int(index)] for index in idx])

    def transform(self, store: ArtifactStore, *, rng: np.random.Generator) -> dict[str, Any]:
        if self._vec is None:
            raise PreprocessValidationError("CountVectorizerStep.transform called before fit()")
        matrix = self._vec.transform(
            _texts(store.require("raw.X"), strip_html=bool(self.strip_html))
        )
        if self.dense:
            matrix = matrix.toarray().astype(np.float32, copy=False)
        return {"features.X": matrix}
