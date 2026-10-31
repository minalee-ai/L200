"""Shared pytest fixtures: PDF builders, an isolated store, and a state seeder."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pymupdf
import pytest
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions

from app.library.store import LibraryStore
from app.runtime import set_library_store


class SeedStateAgent(BaseAgent):
    """Writes `state_delta` as the first step of a test pipeline.

    Stands in for upstream steps whose outputs a test wants to control. Required
    for `temp:` keys, which session services drop from `create_session(state=...)`
    and only honour when written by an event inside the invocation.
    """

    state_delta: dict[str, Any]

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        del ctx
        yield Event(
            author=self.name, actions=EventActions(state_delta=self.state_delta)
        )


def build_bookmarked_pdf_bytes() -> bytes:
    """Creates a 3-chapter PDF with embedded TOC bookmarks and distinct concepts."""
    doc = pymupdf.open()

    ch1 = doc.new_page()
    ch1.insert_text(
        (50, 70),
        "Chapter 1: Foundations of Distributed Consensus\n\n"
        "Distributed systems rely on state machine replication to tolerate node crashes. "
        "Every replica executes deterministic commands in the exact same log order; "
        "consequently, non-faulty nodes converge on identical state. "
        "Safety guarantees that two replicas never commit conflicting values at the same index. "
        "Liveness ensures the cluster eventually makes progress once network partitions heal.",
        fontsize=11,
    )

    ch2 = doc.new_page()
    ch2.insert_text(
        (50, 70),
        "Chapter 2: Leader Election and Quorum Leases\n\n"
        "Raft decomposes consensus into leader election, log replication, and safety checks. "
        "Randomized election timeouts prevent split votes from stalling candidate nodes indefinitely. "
        "A candidate must collect votes from a strict majority quorum before appending new log entries. "
        "Heartbeats from the elected leader suppress unnecessary elections across follower nodes.",
        fontsize=11,
    )

    ch3 = doc.new_page()
    ch3.insert_text(
        (50, 70),
        "Chapter 3: Log Compaction via Snapshots\n\n"
        "Unbounded replication logs eventually exhaust disk capacity and slow down crash recovery. "
        "Periodic snapshotting serializes the current application state to durable storage — "
        "allowing replicas to discard committed log prefixes safely. "
        "Lagging followers receive an InstallSnapshot RPC when required log entries were already compacted.",
        fontsize=11,
    )

    doc.set_toc(
        [
            [1, "Chapter 1: Foundations of Distributed Consensus", 1],
            [1, "Chapter 2: Leader Election and Quorum Leases", 2],
            [1, "Chapter 3: Log Compaction via Snapshots", 3],
        ]
    )
    doc.set_metadata({"title": "Distributed Systems Field Guide"})
    payload = doc.tobytes()
    doc.close()
    return payload


def build_font_headed_paper_bytes() -> bytes:
    """Creates a 2-page research paper PDF with font-size headings and no TOC bookmarks."""
    doc = pymupdf.open()

    p1 = doc.new_page()
    p1.insert_text((50, 55), "Attention Mechanisms in Sequence Modeling", fontsize=18)
    p1.insert_text((50, 95), "1. Introduction", fontsize=14)
    p1.insert_text(
        (50, 120),
        "Recurrent architectures process tokens sequentially, which limits parallel training. "
        "Self-attention connects all positions with a constant number of sequential operations.",
        fontsize=10,
    )

    p2 = doc.new_page()
    p2.insert_text((50, 60), "2. Scaled Dot-Product Attention", fontsize=14)
    p2.insert_text(
        (50, 85),
        "Dot products grow large in magnitude for high key dimensions, pushing softmax into tiny gradients. "
        "Dividing by the square root of key dimension stabilizes gradient flow during training.",
        fontsize=10,
    )

    payload = doc.tobytes()
    doc.close()
    return payload


@pytest.fixture
def isolated_store(tmp_path: Path) -> LibraryStore:
    """Provides a hermetic LibraryStore backed by pytest's temporary directory."""
    store = LibraryStore(store_dir=tmp_path / "library_store")
    set_library_store(store)
    yield store
    set_library_store(None)
