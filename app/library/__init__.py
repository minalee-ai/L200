"""Core domain package for PDF hierarchical chunking, storage, search, and rendering."""

from app.library.chunker import HierarchicalPdfChunker
from app.library.companion_renderer import BookCompanionRenderer
from app.library.models import (
    BookChunk,
    BookProgress,
    BookToc,
    ChapterEntry,
    GroundingReview,
    QuizQuestionDraft,
    QuizSetDraft,
    SearchHit,
    TopicMastery,
)
from app.library.search import PhraseBm25Searcher
from app.library.store import LibraryStore

__all__ = [
    "BookChunk",
    "BookCompanionRenderer",
    "BookProgress",
    "BookToc",
    "ChapterEntry",
    "GroundingReview",
    "HierarchicalPdfChunker",
    "LibraryStore",
    "PhraseBm25Searcher",
    "QuizQuestionDraft",
    "QuizSetDraft",
    "SearchHit",
    "TopicMastery",
]
