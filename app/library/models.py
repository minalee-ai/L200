"""Typed domain models for persistent book indexing, reading progress, and quizzes."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field

# Topics answered correctly less often than this are surfaced for review.
WEAK_TOPIC_THRESHOLD_PCT = 70


class ChapterStatus(StrEnum):
    """Reader-facing reading status of a chapter."""

    DONE = "done"
    READING_NOW = "reading_now"
    UP_NEXT = "up_next"


class ChapterEntry(BaseModel):
    """Metadata for a single detected chapter or major section in a book."""

    chapter_id: str
    title: str
    page_start: int
    page_end: int
    chunk_count: int = 0


class BookToc(BaseModel):
    """Table of contents and document-level metadata persisted per indexed PDF."""

    doc_id: str
    title: str
    total_pages: int
    created_at: str
    chapters: list[ChapterEntry]

    @property
    def chapter_ids(self) -> list[str]:
        """Returns chapter identifiers in reading order."""
        return [chapter.chapter_id for chapter in self.chapters]

    @property
    def first_chapter_id(self) -> str | None:
        """Returns the first chapter identifier, or None for an empty book."""
        return self.chapters[0].chapter_id if self.chapters else None

    def has_chapter(self, chapter_id: str) -> bool:
        """Returns True when `chapter_id` exists in this book."""
        return chapter_id in self.chapter_ids


class BookChunk(BaseModel):
    """A phrase-bounded text chunk tagged with chapter and page provenance."""

    chunk_id: str
    doc_id: str
    chapter_id: str
    chapter_title: str
    page_start: int
    page_end: int
    text: str


class SearchHit(BaseModel):
    """A ranked chunk match returned by phrase-aware BM25 search."""

    chunk: BookChunk
    score: float


class TopicMastery(BaseModel):
    """Cumulative quiz accuracy counters for a single study concept."""

    attempts: int = 0
    correct: int = 0

    @property
    def accuracy_pct(self) -> int:
        """Returns integer accuracy percentage from 0 to 100."""
        if self.attempts == 0:
            return 0
        return round((self.correct / self.attempts) * 100)


class BookProgress(BaseModel):
    """Reader bookmark, completed chapters, quiz mastery, and saved chapter digests.

    This is the single source of truth for a reader's progress in one book;
    session state only mirrors it (see `app.state.reader_state_delta`).
    """

    doc_id: str
    title: str
    bookmark_chapter_id: str = Field(
        # Accept the legacy `current_chapter_id` key from progress files written
        # before the bookmark/active-chapter split.
        validation_alias=AliasChoices("bookmark_chapter_id", "current_chapter_id"),
        description="Next unread chapter; where 'next' resumes reading.",
    )
    completed_chapters: list[str] = Field(default_factory=list)
    topic_mastery: dict[str, TopicMastery] = Field(default_factory=dict)
    chapter_digests: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def new(cls, toc: BookToc) -> BookProgress:
        """Creates fresh progress with the bookmark on the book's first chapter."""
        return cls(
            doc_id=toc.doc_id,
            title=toc.title,
            bookmark_chapter_id=toc.first_chapter_id or "",
        )

    @property
    def overall_mastery_pct(self) -> int:
        """Returns aggregate quiz accuracy across all attempted topics."""
        total_attempts = sum(m.attempts for m in self.topic_mastery.values())
        if total_attempts == 0:
            return 0
        total_correct = sum(m.correct for m in self.topic_mastery.values())
        return round((total_correct / total_attempts) * 100)

    @property
    def weak_topics(self) -> list[str]:
        """Returns attempted topics whose accuracy is below the review threshold."""
        return [
            topic
            for topic, stats in self.topic_mastery.items()
            if stats.attempts > 0 and stats.accuracy_pct < WEAK_TOPIC_THRESHOLD_PCT
        ]

    def chapter_status(self, chapter_id: str) -> ChapterStatus:
        """Classifies a chapter as done, at the bookmark, or still ahead."""
        if chapter_id in self.completed_chapters:
            return ChapterStatus.DONE
        if chapter_id == self.bookmark_chapter_id:
            return ChapterStatus.READING_NOW
        return ChapterStatus.UP_NEXT

    def next_unread_chapter_id(self, toc: BookToc, after: str) -> str | None:
        """Returns the first unread chapter following `after`, wrapping to earlier gaps.

        Prefers unread chapters later in the book than `after`; if none remain,
        falls back to the earliest unread chapter. Returns None when every
        chapter is complete.
        """
        chapter_ids = toc.chapter_ids
        start = chapter_ids.index(after) + 1 if after in chapter_ids else 0
        ordered = chapter_ids[start:] + chapter_ids[:start]
        return next(
            (cid for cid in ordered if cid not in self.completed_chapters), None
        )

    def mark_chapter_read(
        self,
        toc: BookToc,
        chapter_id: str,
        *,
        advance_bookmark: bool,
    ) -> str | None:
        """Marks a known chapter completed and optionally re-points the bookmark.

        The bookmark always lands on an *unread* chapter (so re-reading chapter
        one never drags it backwards) and becomes empty once the book is done.

        Returns:
            The next unread chapter after `chapter_id`, or None if none remain.
        """
        if toc.has_chapter(chapter_id) and chapter_id not in self.completed_chapters:
            self.completed_chapters.append(chapter_id)
        next_unread = self.next_unread_chapter_id(toc, after=chapter_id)
        if advance_bookmark:
            self.bookmark_chapter_id = next_unread or ""
        return next_unread

    def save_chapter_digest(
        self, toc: BookToc, chapter_id: str, digest_md: str
    ) -> str | None:
        """Stores a chapter digest, marks the chapter read, and advances the bookmark.

        Returns:
            The next unread chapter after `chapter_id`, or None if none remain.
        """
        self.chapter_digests[chapter_id] = digest_md.strip()
        return self.mark_chapter_read(toc, chapter_id, advance_bookmark=True)

    def record_topic_result(self, topic: str, passed: bool) -> TopicMastery:
        """Adds one graded quiz attempt for `topic` and returns its updated stats."""
        stats = self.topic_mastery.setdefault(topic, TopicMastery())
        stats.attempts += 1
        if passed:
            stats.correct += 1
        return stats

    def record_quiz_results(
        self,
        toc: BookToc,
        chapter_id: str,
        results: Mapping[str, bool],
    ) -> str | None:
        """Records a whole check-in at once: one attempt per topic, then the bookmark.

        Args:
            toc: Book the quiz belongs to.
            chapter_id: Chapter the quiz covered.
            results: Topic name to whether the reader's answer passed.

        Returns:
            The next unread chapter after `chapter_id`, or None if none remain.
        """
        for topic, passed in results.items():
            self.record_topic_result(topic, passed)
        all_passed = bool(results) and all(results.values())
        return self.mark_chapter_read(toc, chapter_id, advance_bookmark=all_passed)

    def bookshelf_entry(self, toc: BookToc) -> dict[str, Any]:
        """Returns the compact bookshelf summary shown to the reader for this book."""
        return {
            "doc_id": toc.doc_id,
            "title": toc.title,
            "total_pages": toc.total_pages,
            "total_chapters": len(toc.chapters),
            "bookmark_chapter_id": self.bookmark_chapter_id,
            "completed_chapters_count": len(self.completed_chapters),
            "overall_mastery_pct": self.overall_mastery_pct,
        }


class QuizQuestionDraft(BaseModel):
    """A single comprehension question paired with its grounded answer key."""

    question_number: int = Field(description="1-based index of the question.")
    topic: str = Field(description="Concise concept name tested by this question.")
    question_text: str = Field(
        description="Clear comprehension question that does not reveal the answer."
    )
    options: list[str] = Field(
        default_factory=list,
        description="Optional multiple-choice options (e.g., 'A) ...'); empty for open-ended.",
    )
    correct_answer: str = Field(
        description="Expected answer kept hidden until the reader responds."
    )
    supporting_quote: str = Field(
        description="Exact phrase from the chapter chunks proving the answer."
    )
    page_number: int = Field(
        description="1-based PDF page number where supporting_quote appears."
    )


class QuizSetDraft(BaseModel):
    """Structured quiz draft produced by quiz_drafter inside the verification loop."""

    chapter_id: str
    chapter_title: str
    questions: list[QuizQuestionDraft]


class GroundingReview(BaseModel):
    """Verdict emitted by grounding_critic after checking quotes and anti-spoiler rules."""

    grade: Literal["pass", "fail"] = Field(
        description="'pass' only when every supporting_quote is grounded in the chapter text and no question leaks its answer."
    )
    feedback: str = Field(
        description="Specific issues to fix when grade is 'fail', or brief confirmation when 'pass'."
    )


class QuestionGrade(BaseModel):
    """Verdict for one answered quiz question."""

    question_number: int = Field(description="1-based index of the graded question.")
    passed: bool = Field(
        description="True if the reader's answer demonstrates understanding of correct_answer."
    )
    feedback: str = Field(
        description="One or two encouraging sentences explaining the verdict."
    )


class QuizGradeReport(BaseModel):
    """Structured grading output produced by quiz_grader."""

    grades: list[QuestionGrade]
