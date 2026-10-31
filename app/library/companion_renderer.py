"""Renders a book-styled HTML reading companion for the ADK Web Artifacts pane."""

from __future__ import annotations

import html
import re

from app.library.models import (
    WEAK_TOPIC_THRESHOLD_PCT,
    BookProgress,
    BookToc,
    ChapterStatus,
)

_STATUS_BADGES = {
    ChapterStatus.DONE: '<span class="badge done">Read</span>',
    ChapterStatus.READING_NOW: '<span class="badge active">Reading Now</span>',
    ChapterStatus.UP_NEXT: '<span class="badge upcoming">Up Next</span>',
}
_INLINE_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_PAGE_CITE_RE = re.compile(r"\[([^\]]*p\.\s*\d+[^\]]*)\]")


class BookCompanionRenderer:
    """Builds a self-contained HTML abridged-book edition from reader progress."""

    def render_html(self, toc: BookToc, progress: BookProgress) -> str:
        """Returns standalone HTML document displaying TOC, mastery, and read chapters."""
        total_chapters = max(1, len(toc.chapters))
        completed_set = set(progress.completed_chapters)
        read_pct = round((len(completed_set) / total_chapters) * 100)

        toc_items_html = self._render_toc_items(toc, progress)
        mastery_html = self._render_mastery_chips(progress)
        chapters_html = self._render_chapters(toc, progress, completed_set)

        escaped_title = html.escape(toc.title)
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{escaped_title} — BookCompanion Edition</title>
<style>
  :root {{
    --paper: #fbf9f5;
    --card: #ffffff;
    --ink: #1f2421;
    --muted: #5c6460;
    --accent: #2b59c3;
    --accent-soft: #eef3fc;
    --done: #1e7e48;
    --done-bg: #e7f6ed;
    --warn: #9a5b13;
    --warn-bg: #fef4e6;
    --border: #e5e0d8;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    padding: 32px 20px 64px;
    background: var(--paper);
    color: var(--ink);
    font-family: Georgia, "Times New Roman", serif;
    line-height: 1.7;
  }}
  .container {{ max-width: 820px; margin: 0 auto; }}
  header.book-cover {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 28px 32px;
    margin-bottom: 24px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
  }}
  .eyebrow {{
    font-family: system-ui, -apple-system, sans-serif;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--accent);
    font-weight: 700;
  }}
  h1 {{ margin: 8px 0 12px; font-size: 28px; line-height: 1.25; }}
  .stats-row {{
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    font-family: system-ui, -apple-system, sans-serif;
    font-size: 13px;
    color: var(--muted);
    margin-bottom: 14px;
  }}
  .progress-track {{
    height: 8px;
    background: #ece8e1;
    border-radius: 999px;
    overflow: hidden;
  }}
  .progress-fill {{
    height: 100%;
    width: {read_pct}%;
    background: var(--accent);
  }}
  section.panel {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 22px 28px;
    margin-bottom: 24px;
  }}
  h2.panel-title {{
    margin: 0 0 14px;
    font-family: system-ui, -apple-system, sans-serif;
    font-size: 15px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--muted);
  }}
  ul.toc-list {{ list-style: none; padding: 0; margin: 0; }}
  ul.toc-list li {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid #f1ede6;
    font-family: system-ui, -apple-system, sans-serif;
    font-size: 14px;
  }}
  ul.toc-list li:last-child {{ border-bottom: none; }}
  ul.toc-list a {{ color: var(--ink); text-decoration: none; font-weight: 500; }}
  ul.toc-list a:hover {{ color: var(--accent); }}
  .badge {{
    font-family: system-ui, -apple-system, sans-serif;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 9px;
    border-radius: 999px;
  }}
  .badge.done {{ background: var(--done-bg); color: var(--done); }}
  .badge.active {{ background: var(--accent-soft); color: var(--accent); }}
  .badge.upcoming {{ background: #f2efe9; color: var(--muted); }}
  .chips {{ display: flex; flex-wrap: wrap; gap: 8px; }}
  .chip {{
    font-family: system-ui, -apple-system, sans-serif;
    font-size: 12px;
    padding: 4px 10px;
    border-radius: 999px;
  }}
  .chip.strong {{ background: var(--done-bg); color: var(--done); }}
  .chip.review {{ background: var(--warn-bg); color: var(--warn); }}
  article.chapter-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 28px 34px;
    margin-bottom: 24px;
  }}
  article.chapter-card h2 {{ margin-top: 0; font-size: 22px; }}
  article.chapter-card h3 {{ font-size: 18px; margin-top: 20px; }}
  blockquote {{
    margin: 16px 0;
    padding: 12px 18px;
    border-left: 4px solid var(--accent);
    background: var(--accent-soft);
    border-radius: 0 8px 8px 0;
    font-style: italic;
  }}
  .cite {{
    font-family: system-ui, -apple-system, sans-serif;
    font-size: 12px;
    font-style: normal;
    font-weight: 600;
    color: var(--accent);
    background: #e6eefc;
    padding: 1px 6px;
    border-radius: 4px;
  }}
  code {{
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.9em;
    background: #f3f0ea;
    padding: 1px 5px;
    border-radius: 4px;
  }}
  .empty-note {{ color: var(--muted); font-style: italic; }}
</style>
</head>
<body>
  <div class="container">
    <header class="book-cover">
      <div class="eyebrow">Abridged Reading Edition</div>
      <h1>{escaped_title}</h1>
      <div class="stats-row">
        <span><strong>{toc.total_pages}</strong> source pages</span>
        <span><strong>{len(completed_set)} / {len(toc.chapters)}</strong> chapters summarized ({read_pct}%)</span>
        <span><strong>{progress.overall_mastery_pct}%</strong> quiz mastery</span>
      </div>
      <div class="progress-track"><div class="progress-fill"></div></div>
    </header>

    <section class="panel">
      <h2 class="panel-title">Table of Contents &amp; Reading Bookmark</h2>
      <ul class="toc-list">
        {toc_items_html}
      </ul>
    </section>

    <section class="panel">
      <h2 class="panel-title">Concept Mastery</h2>
      {mastery_html}
    </section>

    {chapters_html}
  </div>
</body>
</html>
"""

    def _render_toc_items(self, toc: BookToc, progress: BookProgress) -> str:
        rows: list[str] = []
        for chapter in toc.chapters:
            cid = html.escape(chapter.chapter_id)
            title = html.escape(chapter.title)
            pages = f"pp. {chapter.page_start}&ndash;{chapter.page_end}"
            badge = _STATUS_BADGES[progress.chapter_status(chapter.chapter_id)]
            rows.append(
                f'<li><a href="#{cid}">{title} <span style="color:#5c6460;font-weight:400;">({pages})</span></a>{badge}</li>'
            )
        return "\n        ".join(rows)

    def _render_mastery_chips(self, progress: BookProgress) -> str:
        if not progress.topic_mastery:
            return '<p class="empty-note">Complete a chapter check-in quiz to track concept mastery here.</p>'

        chips: list[str] = []
        for topic, stats in sorted(progress.topic_mastery.items()):
            pct = stats.accuracy_pct
            cls = "strong" if pct >= WEAK_TOPIC_THRESHOLD_PCT else "review"
            label = f"{html.escape(topic)}: {pct}% ({stats.correct}/{stats.attempts})"
            chips.append(f'<span class="chip {cls}">{label}</span>')
        return f'<div class="chips">{"".join(chips)}</div>'

    def _render_chapters(
        self,
        toc: BookToc,
        progress: BookProgress,
        completed_set: set[str],
    ) -> str:
        cards: list[str] = []
        for chapter in toc.chapters:
            digest_md = progress.chapter_digests.get(chapter.chapter_id)
            if not digest_md and chapter.chapter_id not in completed_set:
                continue
            body_html = (
                self._markdown_to_html(digest_md)
                if digest_md
                else '<p class="empty-note">Chapter marked complete.</p>'
            )
            cid = html.escape(chapter.chapter_id)
            title = html.escape(chapter.title)
            cards.append(
                f'<article class="chapter-card" id="{cid}">\n'
                f'  <div class="eyebrow">Chapter {cid.upper()} &middot; Pages {chapter.page_start}&ndash;{chapter.page_end}</div>\n'
                f"  <h2>{title}</h2>\n"
                f"  {body_html}\n"
                f"</article>"
            )

        if not cards:
            return (
                '<section class="panel">'
                '<p class="empty-note">Your abridged chapters will appear here as we read through the book together.</p>'
                "</section>"
            )
        return "\n\n    ".join(cards)

    def _markdown_to_html(self, markdown_text: str) -> str:
        """Converts common Markdown blocks used in chapter digests into safe HTML."""
        lines = markdown_text.splitlines()
        output: list[str] = []
        in_list = False

        for raw_line in lines:
            stripped = raw_line.strip()
            if not stripped:
                if in_list:
                    output.append("</ul>")
                    in_list = False
                continue

            if stripped.startswith("### "):
                if in_list:
                    output.append("</ul>")
                    in_list = False
                output.append(f"<h3>{self._format_inline(stripped[4:])}</h3>")
            elif stripped.startswith("## "):
                if in_list:
                    output.append("</ul>")
                    in_list = False
                output.append(f"<h3>{self._format_inline(stripped[3:])}</h3>")
            elif stripped.startswith("> "):
                if in_list:
                    output.append("</ul>")
                    in_list = False
                output.append(
                    f"<blockquote>{self._format_inline(stripped[2:])}</blockquote>"
                )
            elif stripped.startswith(("- ", "* ")):
                if not in_list:
                    output.append("<ul>")
                    in_list = True
                output.append(f"<li>{self._format_inline(stripped[2:])}</li>")
            else:
                if in_list:
                    output.append("</ul>")
                    in_list = False
                output.append(f"<p>{self._format_inline(stripped)}</p>")

        if in_list:
            output.append("</ul>")
        return "\n".join(output)

    def _format_inline(self, text: str) -> str:
        escaped = html.escape(text)
        escaped = _INLINE_BOLD_RE.sub(r"<strong>\1</strong>", escaped)
        escaped = _INLINE_CODE_RE.sub(r"<code>\1</code>", escaped)
        return _PAGE_CITE_RE.sub(r'<span class="cite">[\1]</span>', escaped)
