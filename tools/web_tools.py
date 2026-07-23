"""Web search, HTTP, PDF, GitHub, and web-scraping tools for every agent.

All agents get these tools automatically via make_all_tools. Every resource
fetched or scraped is chunked and stored in the shared SharedMemoryStore (RAG),
making it instantly available to every other agent via search_web_resources.

Capabilities:
  - web_search            DuckDuckGo search (no API key)
  - github_search         Search GitHub repos, code, and topics
  - fetch_and_store_url   Universal fetcher: HTML, PDF, GitHub repo/file, arXiv, plain text
  - http_request          Raw curl-style HTTP (GET/POST/PUT/etc.) with optional storage
  - search_web_resources  Semantic search over all ingested web content (RAG)
  - list_web_resources    List every URL already ingested by any agent

PDF parsing uses the first available library in priority order:
  pypdf → pdfminer.six → pdftotext CLI → raw text extraction
Install at least one: ``uv add pypdf`` or ``uv add pdfminer.six``

For richer web search install: ``uv add duckduckgo-search``
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool, tool

if TYPE_CHECKING:
    from storage.memory_store import SharedMemoryStore

logger = logging.getLogger(__name__)

# ── Tuning constants ───────────────────────────────────────────────────────

_TIMEOUT        = 20    # seconds per HTTP request
_CHUNK_SIZE     = 2400  # characters per RAG memory chunk
_CHUNK_OVERLAP  = 150   # overlap between adjacent chunks
_MAX_CHUNKS     = 50    # hard ceiling: ≈ 120k chars max per resource
_MAX_PDF_PAGES  = 60    # PDF pages to parse before truncating

# ── HTTP headers ───────────────────────────────────────────────────────────

_BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

_HTML_HDRS = {
    "User-Agent":      _BROWSER_UA,
    "Accept":          "text/html,application/xhtml+xml,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

_GH_HDRS = {
    "User-Agent": "MultiAgentResearcher/1.0",
    "Accept":     "application/vnd.github.v3+json",
}


# ═══════════════════════════════════════════════════════════════════════════
# HTTP layer
# ═══════════════════════════════════════════════════════════════════════════

def _http_raw(
    url: str,
    *,
    method: str = "GET",
    headers: dict | None = None,
    body: bytes | None = None,
) -> tuple[int, dict, bytes]:
    """Low-level HTTP request. Returns (status, response_headers, body_bytes)."""
    merged = {**_HTML_HDRS, **(headers or {})}
    req = urllib.request.Request(url, data=body, headers=merged, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            resp_headers = dict(resp.headers)
            return resp.status, resp_headers, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, {}, exc.read() or b""


def _http_text(url: str, headers: dict | None = None) -> tuple[int, str, str]:
    """Fetch url, return (status, content_type, decoded_text)."""
    status, resp_hdrs, raw = _http_raw(url, headers=headers)
    ct = resp_hdrs.get("Content-Type", "")
    m = re.search(r"charset=([^\s;\"']+)", ct)
    charset = m.group(1).strip('"\'') if m else "utf-8"
    return status, ct, raw.decode(charset, errors="replace")


def _http_json(url: str, headers: dict | None = None) -> tuple[int, Any]:
    """Fetch JSON from url. Returns (status, parsed_object)."""
    merged = {**_GH_HDRS, **(headers or {})}
    status, _ct, text = _http_text(url, headers=merged)
    try:
        return status, json.loads(text)
    except Exception:
        return status, {}


# ═══════════════════════════════════════════════════════════════════════════
# PDF parsing  (multi-fallback)
# ═══════════════════════════════════════════════════════════════════════════

def _pdf_via_pypdf(data: bytes) -> str | None:
    try:
        import pypdf  # type: ignore[import]
        reader = pypdf.PdfReader(io.BytesIO(data))
        pages = reader.pages[:_MAX_PDF_PAGES]
        parts = [p.extract_text() or "" for p in pages]
        return "\n\n".join(parts).strip() or None
    except ImportError:
        return None
    except Exception as exc:
        logger.debug("pypdf failed: %s", exc)
        return None


def _pdf_via_pdfminer(data: bytes) -> str | None:
    try:
        from pdfminer.high_level import extract_text_to_fp  # type: ignore[import]
        from pdfminer.layout import LAParams

        out = io.StringIO()
        extract_text_to_fp(io.BytesIO(data), out, laparams=LAParams(), maxpages=_MAX_PDF_PAGES)
        text = out.getvalue().strip()
        return text or None
    except ImportError:
        return None
    except Exception as exc:
        logger.debug("pdfminer failed: %s", exc)
        return None


def _pdf_via_cli(data: bytes) -> str | None:
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        result = subprocess.run(
            ["pdftotext", "-q", tmp_path, "-"],
            capture_output=True, text=True, timeout=30,
        )
        os.unlink(tmp_path)
        if result.returncode == 0:
            return result.stdout.strip() or None
    except Exception as exc:
        logger.debug("pdftotext CLI failed: %s", exc)
    return None


def _parse_pdf(data: bytes) -> str:
    """Extract text from PDF bytes using the first available method."""
    for fn in (_pdf_via_pypdf, _pdf_via_pdfminer, _pdf_via_cli):
        result = fn(data)
        if result:
            return result
    return (
        "[PDF text extraction failed. Install one of: pypdf, pdfminer.six, "
        "or poppler-utils (pdftotext CLI) for PDF support. "
        f"PDF size: {len(data):,} bytes]"
    )


def _is_pdf(ct: str, url: str) -> bool:
    return "pdf" in ct.lower() or url.lower().split("?")[0].endswith(".pdf")


# ═══════════════════════════════════════════════════════════════════════════
# HTML → readable text extraction
# ═══════════════════════════════════════════════════════════════════════════

_SKIP_TAGS = frozenset({
    "script", "style", "head", "nav", "footer", "form",
    "noscript", "aside", "header", "menu", "advertisement",
})
_BLOCK_TAGS = frozenset({
    "p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "dt", "dd",
    "blockquote", "pre", "div", "section", "article", "main",
    "tr", "th", "td", "caption",
})
_CODE_TAGS = frozenset({"code", "pre", "kbd", "samp"})


class _ContentExtractor(HTMLParser):
    """HTML parser that extracts readable structured text including code blocks."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._buf:      list[str] = []
        self._skip:     int = 0   # depth inside skip-tags
        self._in_code:  int = 0   # depth inside code/pre tags
        self._cur_tag:  str = ""

    def handle_starttag(self, tag: str, attrs: list) -> None:
        self._cur_tag = tag
        if tag in _SKIP_TAGS:
            self._skip += 1
            return
        if tag in _CODE_TAGS:
            self._in_code += 1
            if tag == "pre":
                self._buf.append("\n```\n")
        elif tag in _BLOCK_TAGS and self._buf and not self._buf[-1].endswith("\n"):
            self._buf.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip > 0:
            self._skip -= 1
            return
        if tag in _CODE_TAGS:
            if self._in_code > 0:
                self._in_code -= 1
            if tag == "pre":
                self._buf.append("\n```\n")
        elif tag in _BLOCK_TAGS:
            self._buf.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if self._in_code:
            self._buf.append(data)
        else:
            s = data.strip()
            if s:
                self._buf.append(s + " ")

    def get_text(self) -> str:
        raw = "".join(self._buf)
        # collapse runs of blank lines
        return re.sub(r"\n{3,}", "\n\n", raw).strip()


def _html_to_text(html: str) -> str:
    ex = _ContentExtractor()
    try:
        ex.feed(html)
    except Exception:
        pass
    return ex.get_text()


# ═══════════════════════════════════════════════════════════════════════════
# GitHub helpers
# ═══════════════════════════════════════════════════════════════════════════

def _gh_repo_parts(url: str) -> dict | None:
    """Parse a GitHub URL into its components.

    Returns dict with keys: owner, repo, type (root/file/tree/raw/other),
    branch, path — or None if not a github.com URL.
    """
    m = re.match(
        r"https?://github\.com/([^/]+)/([^/?\s#]+)"
        r"(?:/(blob|tree|raw|releases|issues|pulls|wiki|actions|commits?)?"
        r"(?:/([^/\s?#]+))?"  # branch
        r"(/.+)?)?",          # path
        url,
    )
    if not m:
        # raw.githubusercontent.com
        mr = re.match(
            r"https?://raw\.githubusercontent\.com/([^/]+)/([^/]+)/([^/]+)/(.+)",
            url,
        )
        if mr:
            return {
                "owner": mr.group(1), "repo": mr.group(2),
                "type": "raw", "branch": mr.group(3), "path": mr.group(4),
            }
        return None
    kind = m.group(3) or "root"
    return {
        "owner":  m.group(1),
        "repo":   m.group(2),
        "type":   kind,
        "branch": m.group(4) or "HEAD",
        "path":   (m.group(5) or "").lstrip("/") or "",
    }


def _gh_fetch_readme(owner: str, repo: str) -> str:
    _status, data = _http_json(f"https://api.github.com/repos/{owner}/{repo}/readme")
    if not isinstance(data, dict):
        return ""
    raw_url = data.get("download_url", "")
    if raw_url:
        _s, _ct, text = _http_text(raw_url)
        return text[:8000]
    # fallback: base64 encoded
    content_b64 = data.get("content", "")
    if content_b64:
        return base64.b64decode(content_b64.replace("\n", "")).decode("utf-8", errors="replace")[:8000]
    return ""


def _gh_fetch_file(owner: str, repo: str, path: str, ref: str = "HEAD") -> str:
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    if ref and ref != "HEAD":
        url += f"?ref={ref}"
    _status, data = _http_json(url)
    if not isinstance(data, dict):
        return f"[GitHub: could not fetch {owner}/{repo}/{path}]"
    # single file
    if data.get("encoding") == "base64":
        raw = base64.b64decode(data["content"].replace("\n", ""))
        return raw.decode("utf-8", errors="replace")
    if data.get("download_url"):
        _s, _ct, text = _http_text(data["download_url"])
        return text
    # directory listing
    if isinstance(data, list):
        items = "\n".join(
            f"  {'DIR ' if i.get('type')=='dir' else '    '}{i.get('name','')}"
            for i in data[:60]
        )
        return f"Directory listing ({owner}/{repo}/{path}):\n{items}"
    return f"[GitHub: unexpected response for {path}]"


def _gh_fetch_file_tree(owner: str, repo: str, max_files: int = 80) -> str:
    """Return a flat file tree string using the Git Trees API."""
    _status, data = _http_json(
        f"https://api.github.com/repos/{owner}/{repo}/git/trees/HEAD?recursive=1"
    )
    if not isinstance(data, dict):
        return ""
    tree = data.get("tree", [])
    lines = []
    for item in tree[:max_files]:
        tp = "D" if item.get("type") == "tree" else "F"
        lines.append(f"  [{tp}] {item.get('path', '')}")
    if len(tree) > max_files:
        lines.append(f"  … and {len(tree) - max_files} more files")
    return "\n".join(lines)


def _gh_repo_summary(owner: str, repo: str) -> str:
    """Fetch repo metadata + README + file tree and return a formatted string."""
    _status, meta = _http_json(f"https://api.github.com/repos/{owner}/{repo}")
    parts: list[str] = []

    if isinstance(meta, dict):
        desc    = meta.get("description") or ""
        lang    = meta.get("language") or ""
        stars   = meta.get("stargazers_count", "?")
        topics  = ", ".join(meta.get("topics", []))
        license_name = (meta.get("license") or {}).get("name", "")
        parts.append(
            f"GitHub repo: {owner}/{repo}\n"
            f"Description: {desc}\n"
            f"Language: {lang}   Stars: {stars}   License: {license_name}\n"
            f"Topics: {topics}"
        )

    readme = _gh_fetch_readme(owner, repo)
    if readme:
        parts.append(f"\n--- README ---\n{readme}")

    tree = _gh_fetch_file_tree(owner, repo)
    if tree:
        parts.append(f"\n--- File tree ---\n{tree}")

    return "\n".join(parts)


def _gh_search_repos(query: str, max_results: int = 8) -> list[dict]:
    encoded = urllib.parse.quote_plus(query)
    _status, data = _http_json(
        f"https://api.github.com/search/repositories"
        f"?q={encoded}&per_page={max_results}&sort=stars&order=desc"
    )
    items = data.get("items", []) if isinstance(data, dict) else []
    return [
        {
            "name":        r.get("full_name", ""),
            "url":         r.get("html_url", ""),
            "description": r.get("description", ""),
            "stars":       r.get("stargazers_count", 0),
            "language":    r.get("language", ""),
            "topics":      ", ".join(r.get("topics", [])),
        }
        for r in items
    ]


def _gh_search_code(query: str, max_results: int = 6) -> list[dict]:
    encoded = urllib.parse.quote_plus(query)
    _status, data = _http_json(
        f"https://api.github.com/search/code?q={encoded}&per_page={max_results}"
    )
    items = data.get("items", []) if isinstance(data, dict) else []
    return [
        {
            "name":       r.get("name", ""),
            "path":       r.get("path", ""),
            "repo":       (r.get("repository") or {}).get("full_name", ""),
            "url":        r.get("html_url", ""),
            "raw_url":    r.get("download_url") or r.get("url", ""),
        }
        for r in items
    ]


# ═══════════════════════════════════════════════════════════════════════════
# arXiv helpers
# ═══════════════════════════════════════════════════════════════════════════

def _arxiv_id_from_url(url: str) -> str | None:
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)", url)
    return m.group(1) if m else None


def _arxiv_fetch_meta(arxiv_id: str) -> str:
    """Fetch structured arXiv metadata via the Atom API (no key needed)."""
    clean_id = re.sub(r"v\d+$", "", arxiv_id)
    api_url = f"https://export.arxiv.org/api/query?id_list={clean_id}&max_results=1"
    try:
        _status, _ct, xml = _http_text(api_url)
        # Extract key fields with simple regex (avoid xml.etree for speed)
        title   = re.search(r"<title>(?!ArXiv)(.+?)</title>", xml, re.S)
        summary = re.search(r"<summary>(.+?)</summary>", xml, re.S)
        authors = re.findall(r"<name>(.+?)</name>", xml)
        pub     = re.search(r"<published>(.+?)</published>", xml)
        cats    = re.findall(r'term="([^"]+)"', xml)

        parts = [f"arXiv:{arxiv_id}"]
        if title:
            parts.append(f"Title: {title.group(1).strip()}")
        if authors:
            parts.append(f"Authors: {', '.join(a.strip() for a in authors[:6])}")
        if pub:
            parts.append(f"Published: {pub.group(1)[:10]}")
        if cats:
            parts.append(f"Categories: {', '.join(cats[:5])}")
        if summary:
            parts.append(f"\nAbstract:\n{summary.group(1).strip()}")
        parts.append(f"\nPDF: https://arxiv.org/pdf/{arxiv_id}")
        return "\n".join(parts)
    except Exception as exc:
        logger.debug("arXiv API failed for %s: %s", arxiv_id, exc)
        return f"arXiv:{arxiv_id} (metadata fetch failed)"


# ═══════════════════════════════════════════════════════════════════════════
# DuckDuckGo search
# ═══════════════════════════════════════════════════════════════════════════

def _ddg_search(query: str, max_results: int) -> list[dict]:
    """Return [{title, url, snippet}] via DuckDuckGo."""
    # Preferred: duckduckgo-search library (pip install duckduckgo-search)
    try:
        from duckduckgo_search import DDGS  # type: ignore[import]
        with DDGS() as ddgs:
            return [
                {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
                for r in ddgs.text(query, max_results=max_results)
            ]
    except ImportError:
        pass

    # Fallback: DDG instant-answer JSON API
    enc = urllib.parse.quote_plus(query)
    try:
        _status, _ct, text = _http_text(
            f"https://api.duckduckgo.com/?q={enc}&format=json&no_html=1&skip_disambig=1"
        )
        data = json.loads(text)
        results: list[dict] = []
        if data.get("AbstractURL"):
            results.append({
                "title":   data.get("Heading", query),
                "url":     data.get("AbstractURL", ""),
                "snippet": data.get("Abstract", ""),
            })
        for item in data.get("RelatedTopics", []):
            if isinstance(item, dict) and item.get("FirstURL"):
                results.append({
                    "title":   item.get("Text", "")[:100],
                    "url":     item.get("FirstURL", ""),
                    "snippet": item.get("Text", ""),
                })
        return results[:max_results]
    except Exception as exc:
        logger.warning("DDG search fallback failed: %s", exc)
        return []


# ═══════════════════════════════════════════════════════════════════════════
# Chunking + shared-memory storage
# ═══════════════════════════════════════════════════════════════════════════

def _chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks suitable for vector search."""
    if len(text) <= _CHUNK_SIZE:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text) and len(chunks) < _MAX_CHUNKS:
        end = start + _CHUNK_SIZE
        chunks.append(text[start:end])
        start += _CHUNK_SIZE - _CHUNK_OVERLAP
    return chunks


def _store_resource(
    store: "SharedMemoryStore",
    *,
    url: str,
    content: str,
    label: str,
    agent_name: str,
    source_type: str,          # "html", "pdf", "github", "arxiv", "raw", "api"
) -> list[str]:
    """Chunk content and persist every chunk to the shared memory store.

    Returns list of memory IDs (one per chunk).
    Each chunk carries the source URL and type so semantic search can
    surface the right pieces across large documents.
    """
    chunks = _chunk_text(content.strip())
    total  = len(chunks)
    mem_ids: list[str] = []

    for i, chunk in enumerate(chunks):
        header = (
            f"[WEB_RESOURCE | type={source_type}]\n"
            f"Label:  {label}\n"
            f"Source: {url}\n"
            f"Part:   {i + 1}/{total}\n"
            f"---\n"
        )
        mem_id = store.add_memory(
            header + chunk,
            agent_name=agent_name,
            tags=["web_resource", f"source_type:{source_type}", f"url:{url[:80]}"],
        )
        mem_ids.append(mem_id)

    # One note per resource (dedup key = url) — fast lookup
    store.save_note(
        f"web_resource:{url[:100]}",
        f"[{source_type}] {label[:160]} | {total} chunk(s) | by {agent_name or 'agent'}",
        agent_name=agent_name,
    )
    return mem_ids


# ═══════════════════════════════════════════════════════════════════════════
# Universal URL fetcher  (auto-detects content type and source)
# ═══════════════════════════════════════════════════════════════════════════

def _fetch_url_content(url: str) -> tuple[str, str]:
    """Fetch url and return (source_type, content_text).

    source_type is one of: "github", "arxiv", "pdf", "html", "raw"
    """
    # ── GitHub ─────────────────────────────────────────────────────────────
    gh = _gh_repo_parts(url)
    if gh:
        owner, repo = gh["owner"], gh["repo"]
        kind = gh["type"]
        path = gh["path"]
        ref  = gh["branch"]

        if kind in ("root", ""):
            return "github", _gh_repo_summary(owner, repo)

        if kind in ("blob", "raw") and path:
            content = _gh_fetch_file(owner, repo, path, ref)
            return "github", f"GitHub file: {owner}/{repo}/{path}\n\n{content}"

        if kind == "tree" and path:
            content = _gh_fetch_file(owner, repo, path, ref)
            return "github", f"GitHub dir: {owner}/{repo}/{path}\n\n{content}"

        # Other GitHub page types — fall through to HTML
    else:
        gh = None

    # ── arXiv ──────────────────────────────────────────────────────────────
    arxiv_id = _arxiv_id_from_url(url)
    if arxiv_id:
        meta = _arxiv_fetch_meta(arxiv_id)
        # Also try to get the PDF for full text
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
        try:
            _status, _ct, pdf_bytes = _http_raw(pdf_url)
            if pdf_bytes and len(pdf_bytes) > 1000:
                pdf_text = _parse_pdf(pdf_bytes)
                if pdf_text and not pdf_text.startswith("[PDF text"):
                    return "arxiv", f"{meta}\n\n--- Full text ---\n{pdf_text}"
        except Exception:
            pass
        return "arxiv", meta

    # ── HEAD request to detect content-type cheaply ─────────────────────
    try:
        head_status, head_hdrs, _ = _http_raw(url, method="HEAD")
        ct_head = head_hdrs.get("Content-Type", "")
    except Exception:
        ct_head = ""

    if _is_pdf(ct_head, url):
        # ── PDF ────────────────────────────────────────────────────────────
        _status, ct, pdf_bytes = _http_raw(url)
        return "pdf", _parse_pdf(pdf_bytes)

    # ── Generic HTML / plain-text ──────────────────────────────────────────
    _status, ct, raw = _http_raw(url)
    if _is_pdf(ct, url):
        return "pdf", _parse_pdf(raw)

    ct_lower = ct.lower()
    if "text/plain" in ct_lower or "text/markdown" in ct_lower:
        return "raw", raw.decode("utf-8", errors="replace")

    if "json" in ct_lower or "xml" in ct_lower:
        return "raw", raw.decode("utf-8", errors="replace")[:12000]

    text = _html_to_text(raw.decode("utf-8", errors="replace"))
    return "html", text


# ═══════════════════════════════════════════════════════════════════════════
# Tool factory  — call once per agent, pass the shared store
# ═══════════════════════════════════════════════════════════════════════════

# used in type hints inside helpers above
from typing import Any  # noqa: E402 (after TYPE_CHECKING block)


def make_web_tools(
    store: "SharedMemoryStore",
    agent_name: str = "",
) -> list[BaseTool]:
    """Build web tools bound to the shared memory store.

    Every piece of content fetched or parsed is stored in ``store``
    and is instantly searchable by any other agent sharing the same store.

    Args:
        store: The shared LanceDB-backed memory store.
        agent_name: Agent name stamped on every memory write.
    """

    # ──────────────────────────────────────────────────────────────────────
    @tool
    def web_search(query: str, max_results: int = 6) -> str:
        """Search the web for articles, GitHub repos, documentation, or research papers.

        Returns titles, URLs, and snippets. Use the results to decide which URLs
        to ingest with fetch_and_store_url. Tip: be specific — e.g. 'FastAPI
        async background tasks Python 3.12' beats 'FastAPI'.

        Install ``duckduckgo-search`` for much better results (uv add duckduckgo-search).

        Args:
            query:       Natural-language search query.
            max_results: Results to return (default 6, max 10).
        """
        max_results = min(max_results, 10)
        try:
            results = _ddg_search(query, max_results)
        except Exception as exc:
            return f"Search error: {exc}"
        if not results:
            return f"No results found for: {query!r}"
        lines = [f"Search results for: {query!r}\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['title']}")
            lines.append(f"   URL: {r['url']}")
            if r["snippet"]:
                lines.append(f"   {r['snippet'][:200]}")
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────────
    @tool
    def github_search(query: str, search_type: str = "repositories", max_results: int = 6) -> str:
        """Search GitHub for repositories or code files.

        Useful for finding reference implementations, popular libraries,
        or code examples for a specific pattern.

        Args:
            query:       Search query (e.g. 'FastAPI CRUD SQLite stars:>100').
            search_type: One of 'repositories' (default) or 'code'.
            max_results: Results to return (default 6, max 10).
        """
        max_results = min(max_results, 10)
        try:
            if search_type == "code":
                items = _gh_search_code(query, max_results)
                if not items:
                    return f"No code results for: {query!r}"
                lines = [f"GitHub code search — {query!r}:\n"]
                for i, r in enumerate(items, 1):
                    lines.append(f"{i}. {r['repo']}/{r['path']}")
                    lines.append(f"   URL: {r['url']}")
            else:
                items = _gh_search_repos(query, max_results)
                if not items:
                    return f"No repository results for: {query!r}"
                lines = [f"GitHub repo search — {query!r}:\n"]
                for i, r in enumerate(items, 1):
                    desc = r["description"][:100] if r["description"] else ""
                    lines.append(
                        f"{i}. ⭐{r['stars']:,}  {r['name']}  [{r['language']}]"
                    )
                    if desc:
                        lines.append(f"   {desc}")
                    if r["topics"]:
                        lines.append(f"   Topics: {r['topics']}")
                    lines.append(f"   URL: {r['url']}")
            lines.append(
                "\nTip: call fetch_and_store_url on any URL above to ingest its full content."
            )
            return "\n".join(lines)
        except Exception as exc:
            return f"GitHub search error: {exc}"

    # ──────────────────────────────────────────────────────────────────────
    @tool
    def fetch_and_store_url(url: str, description: str = "") -> str:
        """Fetch ANY URL and store its parsed content in the shared RAG memory.

        Automatically handles:
        • GitHub repos     — README + file tree + repo metadata
        • GitHub files     — raw source code (blob/tree/raw URLs)
        • PDFs             — full text extraction (needs pypdf or pdfminer.six)
        • arXiv papers     — abstract + full text from PDF when available
        • HTML pages       — content extraction (strips nav/ads/scripts)
        • Documentation    — readthedocs, GitHub Pages, official docs
        • Plain text/JSON  — stored as-is
        • REST API URLs    — raw response stored for later reference

        Large documents are automatically chunked so every section is
        independently searchable. All chunks are immediately visible to
        every agent sharing the same memory store.

        Args:
            url:         Full URL to fetch (http or https).
            description: Why this resource matters — stored as context.
                         E.g. 'FastAPI async patterns reference'.
        """
        try:
            source_type, content = _fetch_url_content(url)
        except urllib.error.HTTPError as exc:
            return f"HTTP {exc.code} {exc.reason} — {url}"
        except urllib.error.URLError as exc:
            return f"Network error ({exc.reason}) — {url}"
        except Exception as exc:
            return f"Fetch error: {exc} — {url}"

        content = content.strip()
        if not content:
            return f"Fetched {url} but found no readable content (type: {source_type})."

        label = description.strip() or url
        mem_ids = _store_resource(
            store,
            url=url,
            content=content,
            label=label,
            agent_name=agent_name,
            source_type=source_type,
        )
        preview = content[:300].replace("\n", " ")
        return (
            f"Stored [{source_type}] '{label}'\n"
            f"Chunks: {len(mem_ids)} | First id: {mem_ids[0][:8]}…\n"
            f"Preview: {preview}…\n"
            f"Retrieve with: search_web_resources('{label[:40]}')"
        )

    # ──────────────────────────────────────────────────────────────────────
    @tool
    def http_request(
        method: str,
        url: str,
        headers_json: str = "{}",
        body: str = "",
        store_response: bool = False,
        description: str = "",
    ) -> str:
        """Make a raw HTTP request (like curl) and return the response.

        Use this for REST APIs, webhooks, or any endpoint that requires custom
        headers, methods, or a request body. Optionally store the response in
        shared memory for later retrieval by any agent.

        Args:
            method:         HTTP method — GET, POST, PUT, PATCH, DELETE, HEAD.
            url:            Full URL including query string if needed.
            headers_json:   JSON object of request headers (default: '{}').
                            E.g. '{"Authorization": "Bearer TOKEN", "Content-Type": "application/json"}'
            body:           Request body as a string (for POST/PUT).
                            For JSON APIs pass the JSON string directly.
            store_response: If True, saves the response body to shared memory.
            description:    Label used when store_response=True.
        """
        try:
            extra_headers = json.loads(headers_json) if headers_json.strip() else {}
        except json.JSONDecodeError as exc:
            return f"headers_json parse error: {exc}"

        body_bytes = body.encode("utf-8") if body else None
        try:
            status, resp_hdrs, resp_bytes = _http_raw(
                url, method=method, headers=extra_headers, body=body_bytes
            )
        except urllib.error.URLError as exc:
            return f"Network error: {exc.reason}"
        except Exception as exc:
            return f"Request error: {exc}"

        ct = resp_hdrs.get("Content-Type", "unknown")
        # Try to decode
        try:
            resp_text = resp_bytes.decode("utf-8", errors="replace")
        except Exception:
            resp_text = f"[binary response: {len(resp_bytes)} bytes]"

        result_lines = [
            f"HTTP {status}  {method.upper()} {url}",
            f"Content-Type: {ct}",
            f"Response length: {len(resp_bytes)} bytes",
            "---",
            resp_text[:4000],
        ]
        if len(resp_text) > 4000:
            result_lines.append(f"… [truncated, {len(resp_text) - 4000} more chars]")

        if store_response and resp_text.strip():
            label = description.strip() or f"{method.upper()} {url}"
            _store_resource(
                store,
                url=url,
                content=resp_text,
                label=label,
                agent_name=agent_name,
                source_type="api",
            )
            result_lines.append(f"\n[Response stored in memory as: '{label}']")

        return "\n".join(result_lines)

    # ──────────────────────────────────────────────────────────────────────
    @tool
    def search_web_resources(query: str, limit: int = 5, source_type: str = "") -> str:
        """Semantic search over all web content ingested by any agent.

        Always check this BEFORE fetching a new URL — the resource may already
        be in memory. All agents share the same store, so content fetched by
        a specialist is also visible to the coordinator and vice versa.

        Args:
            query:       What you're looking for (natural language).
            limit:       Max chunks to return (default 5).
            source_type: Optional filter — 'pdf', 'github', 'arxiv', 'html',
                         'api', 'raw'. Leave empty to search all types.
        """
        try:
            # Over-fetch then filter so we always get `limit` web results
            candidates = store.search_memories(
                f"WEB_RESOURCE {query}",
                limit=limit * 6,
            )
            hits = [r for r in candidates if "web_resource" in (r.get("tags") or "")]
            if source_type:
                hits = [r for r in hits if f"source_type:{source_type}" in (r.get("tags") or "")]
            hits = hits[:limit]

            if not hits:
                tip = (
                    f" with source_type='{source_type}'" if source_type else ""
                )
                return (
                    f"No web resources found{tip} matching: {query!r}\n"
                    "Tip: use web_search → fetch_and_store_url to ingest content first."
                )

            lines = [f"Web resources matching {query!r} ({len(hits)} chunks):\n"]
            for r in hits:
                ts = time.strftime("%Y-%m-%d", time.localtime(r.get("created_at", 0)))
                content = r.get("content", "")
                # Show the header + first ~350 chars of the chunk
                lines.append(f"[{ts}]")
                lines.append(content[:450])
                lines.append("---")
            return "\n".join(lines)
        except Exception as exc:
            return f"[WebTools Error] {exc}"

    # ──────────────────────────────────────────────────────────────────────
    @tool
    def list_web_resources(limit: int = 40) -> str:
        """List every URL that has been fetched and stored by any agent.

        Check this before fetching to avoid duplicate work. Shows source type,
        chunk count, and which agent ingested each resource.

        Args:
            limit: Max entries to show (default 40).
        """
        try:
            notes = store.list_notes(prefix="web_resource:")
            if not notes:
                return "No web resources ingested yet. Use fetch_and_store_url to add some."
            shown = notes[:limit]
            lines = [f"Ingested web resources ({len(shown)} of {len(notes)}):\n"]
            for n in shown:
                url_key = n["key"].replace("web_resource:", "")
                desc    = (n.get("value") or "")[:160]
                lines.append(f"  {url_key}")
                lines.append(f"    {desc}")
            return "\n".join(lines)
        except Exception as exc:
            return f"[WebTools Error] {exc}"

    # ──────────────────────────────────────────────────────────────────────
    return [
        web_search,
        github_search,
        fetch_and_store_url,
        http_request,
        search_web_resources,
        list_web_resources,
    ]
