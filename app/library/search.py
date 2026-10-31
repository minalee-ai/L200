"""Phrase-boosted Okapi BM25 search over saved book chunks."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

from app.library.models import BookChunk, SearchHit

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "with",
    }
)


class PhraseBm25Searcher:
    """Ranks chunks using Okapi BM25 term relevance plus contiguous phrase matching."""

    def __init__(
        self, k1: float = 1.5, b: float = 0.75, phrase_boost: float = 2.5
    ) -> None:
        self._k1 = k1
        self._b = b
        self._phrase_boost = phrase_boost

    def search(
        self,
        chunks: Sequence[BookChunk],
        query: str,
        *,
        chapter_id: str = "all",
        limit: int = 5,
    ) -> list[SearchHit]:
        """Returns top matching chunks scoped optionally to a single chapter."""
        scoped_chunks = [
            chunk for chunk in chunks if chapter_id in ("all", "", chunk.chapter_id)
        ]
        if not scoped_chunks or not query.strip():
            return []

        query_terms = self._tokenize(query)
        if not query_terms:
            query_terms = _WORD_RE.findall(query.lower())
        if not query_terms:
            return []

        doc_tokens = [self._tokenize(chunk.text) for chunk in scoped_chunks]
        avg_doc_len = sum(len(tokens) for tokens in doc_tokens) / max(
            1, len(doc_tokens)
        )

        doc_freqs: Counter[str] = Counter()
        for tokens in doc_tokens:
            doc_freqs.update(set(tokens))

        corpus_size = len(scoped_chunks)
        normalized_query_phrase = " ".join(_WORD_RE.findall(query.lower()))
        query_bigrams = [
            f"{query_terms[idx]} {query_terms[idx + 1]}"
            for idx in range(len(query_terms) - 1)
        ]

        hits: list[SearchHit] = []
        for chunk, tokens in zip(scoped_chunks, doc_tokens, strict=True):
            term_counts = Counter(tokens)
            doc_len = len(tokens)
            bm25_score = 0.0

            for term in query_terms:
                freq = term_counts.get(term, 0)
                if freq == 0:
                    continue
                df = doc_freqs[term]
                idf = math.log(1.0 + (corpus_size - df + 0.5) / (df + 0.5))
                denom = freq + self._k1 * (
                    1.0 - self._b + self._b * (doc_len / max(1.0, avg_doc_len))
                )
                bm25_score += idf * ((freq * (self._k1 + 1.0)) / denom)

            normalized_chunk_text = " ".join(_WORD_RE.findall(chunk.text.lower()))
            bonus = 0.0
            if (
                len(query_terms) >= 2
                and normalized_query_phrase in normalized_chunk_text
            ):
                bonus += self._phrase_boost
            elif query_bigrams:
                matched_bigrams = sum(
                    1 for bg in query_bigrams if bg in normalized_chunk_text
                )
                bonus += matched_bigrams * (self._phrase_boost * 0.4)

            total_score = round(bm25_score + bonus, 4)
            if total_score > 0:
                hits.append(SearchHit(chunk=chunk, score=total_score))

        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[: max(1, limit)]

    def _tokenize(self, text: str) -> list[str]:
        return [
            token for token in _WORD_RE.findall(text.lower()) if token not in _STOPWORDS
        ]
