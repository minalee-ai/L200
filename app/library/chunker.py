"""Two-level PDF chunker respecting chapter structure and phrase boundaries."""

from __future__ import annotations

import datetime
import re
import statistics
from dataclasses import dataclass

import pymupdf

from app.library.models import BookChunk, BookToc, ChapterEntry

_HEADING_FONT_RATIO = 1.15
_MAX_HEADING_CHARS = 95
_HEADING_PATTERN = re.compile(
    r"^(?:chapter\s+\d+|section\s+\d+|part\s+[ivxlcdm\d]+|\d+(?:\.\d+)*\.?\s+[A-Z])",
    re.IGNORECASE,
)
# Splits after sentence-ending punctuation or clause delimiters (semicolons / em-dashes)
# while keeping the delimiter attached to the preceding phrase.
_PHRASE_BOUNDARY_RE = re.compile(r"(?<=[.!?;])\s+|(?<=\s—)\s+")


@dataclass(frozen=True)
class _RawChapterSpan:
    """Internal representation of a chapter's title and 1-based inclusive page span."""

    title: str
    page_start: int
    page_end: int


@dataclass(frozen=True)
class _PagePhrase:
    """A single complete sentence or clause paired with its 1-based page number."""

    page_number: int
    text: str
    word_count: int


class HierarchicalPdfChunker:
    """Extracts chapters/sections from a PDF and packs text along phrase boundaries."""

    def __init__(self, target_words: int = 280, overlap_phrases: int = 2) -> None:
        """Initializes chunking parameters.

        Args:
            target_words: Soft upper bound on words per chunk (~370-400 tokens).
            overlap_phrases: Trailing phrases carried into the next chunk for continuity.
        """
        if target_words < 40:
            raise ValueError("target_words must be at least 40.")
        if overlap_phrases < 0:
            raise ValueError("overlap_phrases cannot be negative.")
        self._target_words = target_words
        self._overlap_phrases = overlap_phrases

    def chunk_pdf(
        self,
        pdf_bytes: bytes,
        doc_id: str,
        fallback_title: str,
    ) -> tuple[BookToc, list[BookChunk]]:
        """Parses PDF bytes into a structured BookToc and phrase-safe BookChunk list."""
        if not pdf_bytes:
            raise ValueError("PDF payload is empty.")

        try:
            document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        except Exception as err:
            raise ValueError(f"Invalid or corrupted PDF file: {err}") from err

        with document:
            if document.is_encrypted:
                raise ValueError(
                    "Encrypted PDFs are not supported; unlock the file first."
                )
            if document.page_count == 0:
                raise ValueError("PDF contains zero pages.")

            page_texts = [
                self._normalize_whitespace(document[idx].get_text("text"))
                for idx in range(document.page_count)
            ]
            if not any(page_texts):
                raise ValueError(
                    "PDF has no extractable text (scanned image PDFs are not yet supported)."
                )

            book_title = self._resolve_title(document, fallback_title)
            raw_spans = self._detect_chapters(document, page_texts)

            chapters: list[ChapterEntry] = []
            all_chunks: list[BookChunk] = []

            for index, span in enumerate(raw_spans, start=1):
                chapter_id = f"ch{index:02d}"
                chapter_chunks = self._build_chapter_chunks(
                    doc_id=doc_id,
                    chapter_id=chapter_id,
                    chapter_title=span.title,
                    page_start=span.page_start,
                    page_end=span.page_end,
                    page_texts=page_texts,
                )
                if not chapter_chunks:
                    continue
                chapters.append(
                    ChapterEntry(
                        chapter_id=chapter_id,
                        title=span.title,
                        page_start=span.page_start,
                        page_end=span.page_end,
                        chunk_count=len(chapter_chunks),
                    )
                )
                all_chunks.extend(chapter_chunks)

            toc = BookToc(
                doc_id=doc_id,
                title=book_title,
                total_pages=document.page_count,
                created_at=datetime.datetime.now(datetime.UTC).isoformat(),
                chapters=chapters,
            )
            return toc, all_chunks

    def _resolve_title(self, document: pymupdf.Document, fallback_title: str) -> str:
        metadata = document.metadata or {}
        raw_title = (metadata.get("title") or "").strip()
        if raw_title and raw_title.lower() not in {"untitled", "microsoft word"}:
            return raw_title
        cleaned = (
            fallback_title.removesuffix(".pdf").removesuffix(".PDF").replace("_", " ")
        )
        return cleaned.strip() or "Untitled Book"

    def _detect_chapters(
        self,
        document: pymupdf.Document,
        page_texts: list[str],
    ) -> list[_RawChapterSpan]:
        """Resolves chapter spans via PDF bookmarks first, then font/heading heuristics."""
        total_pages = document.page_count
        toc_entries = document.get_toc(simple=True)
        # Keep top-level (level 1) or section-level (level 2) outline bookmarks.
        top_bookmarks = [
            (title.strip(), max(1, min(page_num, total_pages)))
            for level, title, page_num in toc_entries
            if level in (1, 2) and title.strip()
        ]
        # Prefer level-1 chapters when multiple exist so dense sub-headings don't fragment a book.
        level_one = [
            (title.strip(), max(1, min(page_num, total_pages)))
            for level, title, page_num in toc_entries
            if level == 1 and title.strip()
        ]
        chosen_bookmarks = level_one if len(level_one) >= 2 else top_bookmarks
        if chosen_bookmarks:
            return self._bookmarks_to_spans(chosen_bookmarks, total_pages)

        heading_bookmarks = self._detect_headings_from_fonts(document)
        if heading_bookmarks:
            return self._bookmarks_to_spans(heading_bookmarks, total_pages)

        # Fallback for plain unheaded documents: group into readable ~8-page sections.
        if total_pages <= 8:
            return [
                _RawChapterSpan(title="Full Text", page_start=1, page_end=total_pages)
            ]

        spans: list[_RawChapterSpan] = []
        pages_per_part = 8
        for part_num, start in enumerate(
            range(1, total_pages + 1, pages_per_part), start=1
        ):
            end = min(total_pages, start + pages_per_part - 1)
            if any(page_texts[p - 1] for p in range(start, end + 1)):
                spans.append(
                    _RawChapterSpan(
                        title=f"Part {part_num} (Pages {start}-{end})",
                        page_start=start,
                        page_end=end,
                    )
                )
        return spans

    def _detect_headings_from_fonts(
        self, document: pymupdf.Document
    ) -> list[tuple[str, int]]:
        """Identifies chapter/section headings using font-size ratios and numbering."""
        font_sizes: list[float] = []
        line_records: list[tuple[int, float, str]] = []

        for page_idx in range(document.page_count):
            page_dict = document[page_idx].get_text("dict")
            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    line_text = "".join(s.get("text", "") for s in spans).strip()
                    if not line_text:
                        continue
                    max_size = max(
                        (float(s.get("size", 0.0)) for s in spans), default=0.0
                    )
                    if max_size > 0:
                        # Weight font-size samples by character count so body text sets the median.
                        font_sizes.extend([max_size] * max(1, min(len(line_text), 40)))
                        line_records.append((page_idx + 1, max_size, line_text))

        if not font_sizes:
            return []

        median_size = statistics.median(font_sizes)
        candidates: list[tuple[str, int]] = []
        seen_titles: set[str] = set()

        for page_num, size, text in line_records:
            cleaned = re.sub(r"\s+", " ", text).strip()
            if len(cleaned) < 3 or len(cleaned) > _MAX_HEADING_CHARS:
                continue
            is_larger_font = size >= median_size * _HEADING_FONT_RATIO
            matches_pattern = bool(_HEADING_PATTERN.match(cleaned))
            if not (is_larger_font or matches_pattern):
                continue
            # Skip document title on page 1 unless explicitly numbered like "1. Introduction".
            if page_num == 1 and not matches_pattern and not candidates:
                if size >= median_size * 1.4:
                    continue
            normalized_key = cleaned.lower()
            if normalized_key in seen_titles:
                continue
            seen_titles.add(normalized_key)
            candidates.append((cleaned, page_num))

        return candidates if len(candidates) >= 2 else []

    def _bookmarks_to_spans(
        self,
        bookmarks: list[tuple[str, int]],
        total_pages: int,
    ) -> list[_RawChapterSpan]:
        """Converts (title, start_page) pairs into contiguous non-empty page ranges."""
        deduped: list[tuple[str, int]] = []
        for title, page_num in bookmarks:
            if deduped and deduped[-1][1] == page_num:
                # If a page has a generic banner/title followed by an explicit numbered
                # section (e.g., "1. Introduction"), prefer the numbered section title.
                if _HEADING_PATTERN.match(title) and not _HEADING_PATTERN.match(
                    deduped[-1][0]
                ):
                    deduped[-1] = (title, page_num)
                continue
            deduped.append((title, page_num))

        spans: list[_RawChapterSpan] = []
        # Include any front matter preceding the first detected chapter.
        first_page = deduped[0][1]
        if first_page > 1:
            spans.append(
                _RawChapterSpan(
                    title="Preface & Overview",
                    page_start=1,
                    page_end=first_page - 1,
                )
            )

        for idx, (title, start_page) in enumerate(deduped):
            if idx + 1 < len(deduped):
                next_start = deduped[idx + 1][1]
                # When the next section starts mid-page, include that boundary page in the
                # preceding span only if start_page would otherwise exceed end_page.
                end_page = max(start_page, next_start - 1)
            else:
                end_page = total_pages
            spans.append(
                _RawChapterSpan(title=title, page_start=start_page, page_end=end_page)
            )
        return spans

    def _build_chapter_chunks(
        self,
        *,
        doc_id: str,
        chapter_id: str,
        chapter_title: str,
        page_start: int,
        page_end: int,
        page_texts: list[str],
    ) -> list[BookChunk]:
        """Packs chapter pages into phrase-bounded chunks with sliding overlap."""
        phrases: list[_PagePhrase] = []
        for page_num in range(page_start, page_end + 1):
            raw_page = page_texts[page_num - 1]
            if not raw_page:
                continue
            for phrase_text in self._extract_phrases(raw_page):
                words = len(phrase_text.split())
                if words > 0:
                    phrases.append(
                        _PagePhrase(
                            page_number=page_num,
                            text=phrase_text,
                            word_count=words,
                        )
                    )

        if not phrases:
            return []

        chunks: list[BookChunk] = []
        cursor = 0
        chunk_idx = 1

        while cursor < len(phrases):
            window: list[_PagePhrase] = []
            word_total = 0
            end_cursor = cursor

            while end_cursor < len(phrases):
                candidate = phrases[end_cursor]
                if window and word_total + candidate.word_count > self._target_words:
                    break
                window.append(candidate)
                word_total += candidate.word_count
                end_cursor += 1

            chunks.append(
                BookChunk(
                    chunk_id=f"{chapter_id}_c{chunk_idx:03d}",
                    doc_id=doc_id,
                    chapter_id=chapter_id,
                    chapter_title=chapter_title,
                    page_start=window[0].page_number,
                    page_end=window[-1].page_number,
                    text=" ".join(item.text for item in window),
                )
            )
            chunk_idx += 1

            if end_cursor >= len(phrases):
                break

            # Advance cursor while retaining trailing phrases so cross-sentence references
            # ("This mechanism...", "In contrast...") keep their antecedent in the next chunk.
            step = max(1, len(window) - self._overlap_phrases)
            cursor += step

        return chunks

    def _extract_phrases(self, page_text: str) -> list[str]:
        """Splits page text on paragraph and sentence/clause boundaries without mid-phrase cuts."""
        phrases: list[str] = []
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", page_text) if p.strip()]
        for paragraph in paragraphs:
            single_line = re.sub(r"\s+", " ", paragraph).strip()
            for segment in _PHRASE_BOUNDARY_RE.split(single_line):
                cleaned = segment.strip()
                if not cleaned:
                    continue
                if len(cleaned.split()) <= self._target_words:
                    phrases.append(cleaned)
                else:
                    # Sub-split unusually long run-on sentences at comma clauses rather than raw chars.
                    clauses = [
                        c.strip() for c in re.split(r"(?<=,)\s+", cleaned) if c.strip()
                    ]
                    phrases.extend(clauses)
        return phrases

    def _normalize_whitespace(self, raw_text: str) -> str:
        # Rejoin hyphenated line-wraps common in two-column research papers.
        dehyphenated = re.sub(r"(\w)-\n(\w)", r"\1\2", raw_text)
        return dehyphenated.strip()
