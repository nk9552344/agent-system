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
  pypdf → pdfminer.six → pdftotext CLI
Install at least one: ``uv add pypdf`` or ``uv add pdfminer.six``

For richer web search install: ``uv add duckduckgo-search``

All tunables (GitHub token, timeout, chunk sizes, PDF page limit) are
configured via the ``web:`` section of config.yml and passed in via WebConfig.
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
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool, tool

if TYPE_CHECKING:
    from storage.memory_store import SharedMemoryStore

logger = logging.getLogger(__name__)

_BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


# ═══════════════════════════════════════════════════════════════════════════
# WebConfig  — all user-tunable knobs, read from config.yml → main.py
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class WebConfig:
    """Settings that control web tool behaviour.

    All fields have production-safe defaults so the tools work out of the box.
    Override any value via the ``web:`` section of config.yml — main.py reads
    that section and constructs a WebConfig that is threaded through to every
    agent via make_all_tools / OllamaDeepAgent.

    Attributes:
        github_token:
            GitHub Personal Access Token.
            Without one: 60 API requests / hour (easily exhausted).
            With one:  5 000 API requests / hour.
            Create at https://github.com/settings/tokens
            (no scopes needed for public repos).
            Alternatively set the GITHUB_TOKEN environment variable —
            the token in config.yml always takes precedence.
        timeout:
            HTTP request timeout in seconds (default 20).
            Increase for slow networks or large file downloads.
        max_pdf_pages:
            Maximum PDF pages to extract before truncating (default 60).
            Reduce if PDF parsing is too slow on your hardware.
        chunk_size:
            Characters per RAG memory chunk (default 2 400).
            Smaller → finer-grained retrieval; larger → fewer chunks.
        chunk_overlap:
            Overlap between consecutive chunks in characters (default 150).
            Keeps sentence context intact across chunk boundaries.
        max_chunks_per_resource:
            Hard cap on chunks stored per URL (default 50 ≈ 120 k chars).
            Prevents one huge document from flooding the memory store.
        user_agent:
            HTTP User-Agent header for HTML page requests.
    """

    github_token:            str = field(default="")
    timeout:                 int = field(default=20)
    max_pdf_pages:           int = field(default=60)
    chunk_size:              int = field(default=2400)
    chunk_overlap:           int = field(default=150)
    max_chunks_per_resource: int = field(default=50)
    user_agent:              str = field(default=_BROWSER_UA)

    def __post_init__(self) -> None:
        # GITHUB_TOKEN env var as fallback (token in config.yml wins)
        if not self.github_token:
            self.github_token = os.environ.get("GITHUB_TOKEN", "")

    def gh_headers(self) -> dict:
        """GitHub API headers — includes Bearer auth when a token is configured."""
        hdrs: dict = {
            "User-Agent": "MultiAgentResearcher/1.0",
            "Accept":     "application/vnd.github.v3+json",
        }
        if self.github_token:
            hdrs["Authorization"] = f"Bearer {self.github_token}"
        return hdrs

    def html_headers(self) -> dict:
        return {
            "User-Agent":      self.user_agent,
            "Accept":          "text/html,application/xhtml+xml,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WebConfig":
        """Build a WebConfig from a raw dict (e.g. cfg.get('web', {}))."""
        return cls(
            github_token=            str(d.get("github_token", "") or ""),
            timeout=                 int(d.get("timeout",                 20)),
            max_pdf_pages=           int(d.get("max_pdf_pages",           60)),
            chunk_size=              int(d.get("chunk_size",            2400)),
            chunk_overlap=           int(d.get("chunk_overlap",          150)),
            max_chunks_per_resource= int(d.get("max_chunks_per_resource", 50)),
            user_agent=              str(d.get("user_agent", _BROWSER_UA) or _BROWSER_UA),
        )


# ═══════════════════════════════════════════════════════════════════════════
# HTTP layer
# ═══════════════════════════════════════════════════════════════════════════

def _http_raw(
    url: str,
    cfg: WebConfig,
    *,
    method: str = "GET",
    extra_headers: dict | None = None,
    body: bytes | None = None,
) -> tuple[int, dict, bytes]:
    """Low-level HTTP request.  Returns (status_code, response_headers, body_bytes)."""
    merged = {**cfg.html_headers(), **(extra_headers or {})}
    req = urllib.request.Request(url, data=body, headers=merged, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, {}, exc.read() or b""


def _http_text(url: str, cfg: WebConfig, extra_headers: dict | None = None) -> tuple[int, str, str]:
    """Fetch url → (status, content_type, decoded_text)."""
    status, hdrs, raw = _http_raw(url, cfg, extra_headers=extra_headers)
    ct = hdrs.get("Content-Type", "")
    m = re.search(r"charset=([^\s;\"']+)", ct)
    charset = m.group(1).strip("\"'") if m else "utf-8"
    return status, ct, raw.decode(charset, errors="replace")


def _http_json(url: str, cfg: WebConfig) -> tuple[int, Any]:
    """Fetch JSON from url using GitHub-style headers.  Returns (status, parsed_object)."""
    status, _ct, text = _http_text(url, cfg, extra_headers=cfg.gh_headers())
    try:
        return status, json.loads(text)
    except Exception:
        return status, {}


# ═══════════════════════════════════════════════════════════════════════════
# PDF parsing  (multi-fallback: pypdf → pdfminer.six → pdftotext CLI)
# ═══════════════════════════════════════════════════════════════════════════

def _pdf_via_pypdf(data: bytes, max_pages: int) -> str | None:
    try:
        import pypdf  # type: ignore[import]
        reader = pypdf.PdfReader(io.BytesIO(data))
        texts = [p.extract_text() or "" for p in reader.pages[:max_pages]]
        return "\n\n".join(texts).strip() or None
    except ImportError:
        return None
    except Exception as exc:
        logger.debug("pypdf failed: %s", exc)
        return None


def _pdf_via_pdfminer(data: bytes, max_pages: int) -> str | None:
    try:
        from pdfminer.high_level import extract_text_to_fp  # type: ignore[import]
        from pdfminer.layout import LAParams
        out = io.StringIO()
        extract_text_to_fp(io.BytesIO(data), out, laparams=LAParams(), maxpages=max_pages)
        return out.getvalue().strip() or None
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
            capture_output=True, text=True, timeout=45,
        )
        os.unlink(tmp_path)
        if result.returncode == 0:
            return result.stdout.strip() or None
    except Exception as exc:
        logger.debug("pdftotext CLI failed: %s", exc)
    return None


def _parse_pdf(data: bytes, cfg: WebConfig) -> str:
    """Extract text from PDF bytes using the first available method."""
    for fn in (
        lambda: _pdf_via_pypdf(data, cfg.max_pdf_pages),
        lambda: _pdf_via_pdfminer(data, cfg.max_pdf_pages),
        lambda: _pdf_via_cli(data),
    ):
        result = fn()
        if result:
            return result
    return (
        f"[PDF text extraction failed — {len(data):,} bytes. "
        "Install one of: pypdf, pdfminer.six, or poppler-utils (pdftotext CLI). "
        "Commands: uv add pypdf  OR  uv add pdfminer.six]"
    )


def _is_pdf(ct: str, url: str) -> bool:
    return "pdf" in ct.lower() or url.lower().split("?")[0].endswith(".pdf")


# ═══════════════════════════════════════════════════════════════════════════
# HTML → readable text extraction
# ═══════════════════════════════════════════════════════════════════════════

_SKIP_TAGS = frozenset({
    "script", "style", "head", "nav", "footer", "form",
    "noscript", "aside", "header", "menu",
})
_BLOCK_TAGS = frozenset({
    "p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "dt", "dd",
    "blockquote", "pre", "div", "section", "article", "main",
    "tr", "th", "td", "caption",
})
_CODE_TAGS = frozenset({"code", "pre", "kbd", "samp"})


class _ContentExtractor(HTMLParser):
    """Extract readable text + code blocks from HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._buf:     list[str] = []
        self._skip:    int = 0
        self._in_code: int = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
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
        if tag in _SKIP_TAGS:
            if self._skip > 0:
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

def _gh_parse_url(url: str) -> dict | None:
    """Decompose a GitHub URL into {owner, repo, type, branch, path}.

    Returns None if the URL is not a github.com URL.
    """
    # raw.githubusercontent.com
    mr = re.match(
        r"https?://raw\.githubusercontent\.com/([^/]+)/([^/]+)/([^/]+)/(.+)", url
    )
    if mr:
        return {"owner": mr.group(1), "repo": mr.group(2),
                "type": "raw", "branch": mr.group(3), "path": mr.group(4)}

    m = re.match(
        r"https?://github\.com/([^/]+)/([^/?\s#]+)"
        r"(?:/(blob|tree|raw|releases|issues|pulls|wiki|commits?)?"
        r"(?:/([^/\s?#]+))?"   # branch
        r"(/.+)?)?",            # path
        url,
    )
    if not m:
        return None
    kind = m.group(3) or "root"
    return {
        "owner":  m.group(1),
        "repo":   m.group(2),
        "type":   kind,
        "branch": m.group(4) or "HEAD",
        "path":   (m.group(5) or "").lstrip("/"),
    }


def _gh_fetch_readme(owner: str, repo: str, cfg: WebConfig) -> str:
    _status, data = _http_json(f"https://api.github.com/repos/{owner}/{repo}/readme", cfg)
    if not isinstance(data, dict):
        return ""
    raw_url = data.get("download_url", "")
    if raw_url:
        _s, _ct, text = _http_text(raw_url, cfg)
        return text[:8000]
    content_b64 = data.get("content", "")
    if content_b64:
        return base64.b64decode(content_b64.replace("\n", "")).decode("utf-8", errors="replace")[:8000]
    return ""


def _gh_fetch_file(owner: str, repo: str, path: str, cfg: WebConfig, ref: str = "HEAD") -> str:
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    if ref and ref != "HEAD":
        url += f"?ref={ref}"
    _status, data = _http_json(url, cfg)
    if isinstance(data, list):
        # directory listing
        lines = "\n".join(
            f"  {'DIR' if i.get('type') == 'dir' else 'FILE'}  {i.get('name', '')}"
            for i in data[:80]
        )
        return f"Directory: {owner}/{repo}/{path}\n{lines}"
    if isinstance(data, dict):
        if data.get("encoding") == "base64":
            raw = base64.b64decode(data["content"].replace("\n", ""))
            return raw.decode("utf-8", errors="replace")
        if data.get("download_url"):
            _s, _ct, text = _http_text(data["download_url"], cfg)
            return text
    return f"[GitHub: could not fetch {owner}/{repo}/{path}]"


def _gh_file_tree(owner: str, repo: str, cfg: WebConfig, max_files: int = 100) -> str:
    _status, data = _http_json(
        f"https://api.github.com/repos/{owner}/{repo}/git/trees/HEAD?recursive=1", cfg
    )
    if not isinstance(data, dict):
        return ""
    tree = data.get("tree", [])
    lines = [
        f"  {'D' if i.get('type') == 'tree' else 'F'}  {i.get('path', '')}"
        for i in tree[:max_files]
    ]
    if len(tree) > max_files:
        lines.append(f"  … and {len(tree) - max_files} more")
    return "\n".join(lines)


def _gh_repo_summary(owner: str, repo: str, cfg: WebConfig) -> str:
    _status, meta = _http_json(f"https://api.github.com/repos/{owner}/{repo}", cfg)
    parts: list[str] = []
    if isinstance(meta, dict):
        parts.append(
            f"GitHub repo: {owner}/{repo}\n"
            f"Description: {meta.get('description') or ''}\n"
            f"Language: {meta.get('language') or ''}  "
            f"Stars: {meta.get('stargazers_count', '?')}  "
            f"License: {(meta.get('license') or {}).get('name', '')}\n"
            f"Topics: {', '.join(meta.get('topics', []))}"
        )
    readme = _gh_fetch_readme(owner, repo, cfg)
    if readme:
        parts.append(f"\n--- README ---\n{readme}")
    tree = _gh_file_tree(owner, repo, cfg)
    if tree:
        parts.append(f"\n--- File tree ---\n{tree}")
    return "\n".join(parts)


def _gh_search_repos(query: str, cfg: WebConfig, max_results: int = 8) -> list[dict]:
    enc = urllib.parse.quote_plus(query)
    _status, data = _http_json(
        f"https://api.github.com/search/repositories"
        f"?q={enc}&per_page={max_results}&sort=stars&order=desc",
        cfg,
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


def _gh_search_code(query: str, cfg: WebConfig, max_results: int = 6) -> list[dict]:
    enc = urllib.parse.quote_plus(query)
    _status, data = _http_json(
        f"https://api.github.com/search/code?q={enc}&per_page={max_results}",
        cfg,
    )
    items = data.get("items", []) if isinstance(data, dict) else []
    return [
        {
            "name":    r.get("name", ""),
            "path":    r.get("path", ""),
            "repo":    (r.get("repository") or {}).get("full_name", ""),
            "url":     r.get("html_url", ""),
            "raw_url": r.get("download_url") or r.get("url", ""),
        }
        for r in items
    ]


# ═══════════════════════════════════════════════════════════════════════════
# arXiv helpers
# ═══════════════════════════════════════════════════════════════════════════

def _arxiv_id_from_url(url: str) -> str | None:
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)", url)
    return m.group(1) if m else None


def _arxiv_fetch(arxiv_id: str, cfg: WebConfig) -> str:
    """Fetch arXiv metadata via Atom API, then attempt full-text PDF extraction."""
    clean_id = re.sub(r"v\d+$", "", arxiv_id)
    _status, _ct, xml = _http_text(
        f"https://export.arxiv.org/api/query?id_list={clean_id}&max_results=1", cfg
    )
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
    meta_text = "\n".join(parts)

    # Also extract full text from the PDF
    try:
        _s, _ct, pdf_bytes_raw = _http_raw(f"https://arxiv.org/pdf/{arxiv_id}", cfg)
        if pdf_bytes_raw and len(pdf_bytes_raw) > 4096:
            full_text = _parse_pdf(pdf_bytes_raw, cfg)
            if full_text and not full_text.startswith("[PDF text"):
                return f"{meta_text}\n\n--- Full text ---\n{full_text}"
    except Exception:
        pass
    return meta_text


# ═══════════════════════════════════════════════════════════════════════════
# DuckDuckGo search
# ═══════════════════════════════════════════════════════════════════════════

def _ddg_search(query: str, max_results: int) -> list[dict]:
    """Return [{title, url, snippet}] via DuckDuckGo (no API key required)."""
    try:
        from duckduckgo_search import DDGS  # type: ignore[import]
        with DDGS() as ddgs:
            return [
                {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
                for r in ddgs.text(query, max_results=max_results)
            ]
    except ImportError:
        pass

    enc = urllib.parse.quote_plus(query)
    try:
        # Use a plain urlopen for the fallback (no cfg needed — no auth, short timeout)
        req = urllib.request.Request(
            f"https://api.duckduckgo.com/?q={enc}&format=json&no_html=1&skip_disambig=1",
            headers={"User-Agent": "MultiAgentBot/1.0"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
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

def _chunk_text(text: str, cfg: WebConfig) -> list[str]:
    """Split text into overlapping chunks sized for vector-search retrieval."""
    if len(text) <= cfg.chunk_size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text) and len(chunks) < cfg.max_chunks_per_resource:
        chunks.append(text[start : start + cfg.chunk_size])
        start += cfg.chunk_size - cfg.chunk_overlap
    return chunks


def _store_resource(
    store: "SharedMemoryStore",
    cfg: WebConfig,
    *,
    url: str,
    content: str,
    label: str,
    agent_name: str,
    source_type: str,
) -> list[str]:
    """Chunk content and persist every chunk to the shared memory store.

    Each chunk carries full provenance (source URL, type, part N/M) so the
    LLM always knows where a retrieved snippet came from.  Returns memory IDs.
    """
    chunks = _chunk_text(content.strip(), cfg)
    total  = len(chunks)
    mem_ids: list[str] = []

    for i, chunk in enumerate(chunks):
        header = (
            f"[WEB_RESOURCE | type={source_type}]\n"
            f"Label:  {label}\n"
            f"Source: {url}\n"
            f"Part:   {i + 1}/{total}\n"
            "---\n"
        )
        mem_id = store.add_memory(
            header + chunk,
            agent_name=agent_name,
            tags=["web_resource", f"source_type:{source_type}", f"url:{url[:80]}"],
        )
        mem_ids.append(mem_id)

    # Dedup index — one note per URL, fast exact lookup
    store.save_note(
        f"web_resource:{url[:100]}",
        f"[{source_type}] {label[:160]} | {total} chunk(s) | by {agent_name or 'agent'}",
        agent_name=agent_name,
    )
    return mem_ids


# ═══════════════════════════════════════════════════════════════════════════
# Universal URL fetcher  (auto-detects content type and source)
# ═══════════════════════════════════════════════════════════════════════════

def _fetch_url_content(url: str, cfg: WebConfig) -> tuple[str, str]:
    """Fetch url → (source_type, content_text).

    source_type: "github" | "arxiv" | "pdf" | "html" | "raw"
    """
    # ── GitHub ──────────────────────────────────────────────────────────
    gh = _gh_parse_url(url)
    if gh:
        owner, repo = gh["owner"], gh["repo"]
        kind        = gh["type"]
        path        = gh["path"]
        ref         = gh["branch"]

        if kind in ("root", ""):
            return "github", _gh_repo_summary(owner, repo, cfg)

        if kind in ("blob", "raw") and path:
            content = _gh_fetch_file(owner, repo, path, cfg, ref)
            return "github", f"GitHub file: {owner}/{repo}/{path}\n\n{content}"

        if kind == "tree" and path:
            content = _gh_fetch_file(owner, repo, path, cfg, ref)
            return "github", f"GitHub dir: {owner}/{repo}/{path}\n\n{content}"
        # other GitHub page types fall through to HTML parse

    # ── arXiv ───────────────────────────────────────────────────────────
    arxiv_id = _arxiv_id_from_url(url)
    if arxiv_id:
        return "arxiv", _arxiv_fetch(arxiv_id, cfg)

    # ── HEAD to detect content-type cheaply ─────────────────────────────
    try:
        _hs, head_hdrs, _ = _http_raw(url, cfg, method="HEAD")
        ct_head = head_hdrs.get("Content-Type", "")
    except Exception:
        ct_head = ""

    if _is_pdf(ct_head, url):
        _s, _ct, raw_bytes = _http_raw(url, cfg)
        return "pdf", _parse_pdf(raw_bytes, cfg)

    # ── Full GET ─────────────────────────────────────────────────────────
    _status, ct, raw_text_or_bytes = None, "", b""
    try:
        _status, ct, raw_text_or_bytes = _http_raw(url, cfg)
    except Exception as exc:
        raise

    if _is_pdf(ct, url):
        return "pdf", _parse_pdf(raw_text_or_bytes, cfg)

    decoded = raw_text_or_bytes.decode("utf-8", errors="replace")
    ct_lower = ct.lower()

    if "text/plain" in ct_lower or "text/markdown" in ct_lower:
        return "raw", decoded

    if "json" in ct_lower or "xml" in ct_lower:
        return "raw", decoded[:16000]

    return "html", _html_to_text(decoded)


# ═══════════════════════════════════════════════════════════════════════════
# Tool factory
# ═══════════════════════════════════════════════════════════════════════════

def make_web_tools(
    store: "SharedMemoryStore",
    agent_name: str = "",
    config: WebConfig | None = None,
) -> list[BaseTool]:
    """Build web tools bound to the shared memory store and WebConfig.

    Args:
        store:      The shared LanceDB-backed memory store (all agents share one).
        agent_name: Agent identity stamped on every memory write.
        config:     Web tool settings from config.yml.  Uses safe defaults if None.
    """
    cfg = config or WebConfig()

    # ──────────────────────────────────────────────────────────────────────
    @tool
    def web_search(query: str, max_results: int = 6) -> str:
        """Search the web for articles, GitHub repos, documentation, or research papers.

        Returns titles, URLs, and snippets. Use the results to decide which URLs
        to ingest with fetch_and_store_url.

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

        Set a GitHub token in config.yml (web.github_token) or the GITHUB_TOKEN
        environment variable to raise the API rate limit from 60 to 5 000 req/hr.

        Args:
            query:       Search terms (e.g. 'FastAPI CRUD SQLite stars:>500').
            search_type: 'repositories' (default) or 'code'.
            max_results: Results to return (default 6, max 10).
        """
        max_results = min(max_results, 10)
        try:
            if search_type == "code":
                items = _gh_search_code(query, cfg, max_results)
                if not items:
                    return f"No code results for: {query!r}"
                lines = [f"GitHub code search — {query!r}:\n"]
                for i, r in enumerate(items, 1):
                    lines.append(f"{i}. {r['repo']}/{r['path']}")
                    lines.append(f"   URL: {r['url']}")
            else:
                items = _gh_search_repos(query, cfg, max_results)
                if not items:
                    return f"No repository results for: {query!r}"
                lines = [f"GitHub repo search — {query!r}:\n"]
                for i, r in enumerate(items, 1):
                    desc = (r["description"] or "")[:100]
                    lines.append(f"{i}. ⭐{r['stars']:,}  {r['name']}  [{r['language']}]")
                    if desc:
                        lines.append(f"   {desc}")
                    if r["topics"]:
                        lines.append(f"   Topics: {r['topics']}")
                    lines.append(f"   URL: {r['url']}")
            auth_note = "" if cfg.github_token else " (tip: set web.github_token in config.yml for higher rate limits)"
            lines.append(f"\nTip: call fetch_and_store_url on any URL to ingest its content.{auth_note}")
            return "\n".join(lines)
        except Exception as exc:
            return f"GitHub search error: {exc}"

    # ──────────────────────────────────────────────────────────────────────
    @tool
    def fetch_and_store_url(url: str, description: str = "") -> str:
        """Fetch ANY URL and store its parsed content in the shared RAG memory.

        Automatically handles:
        • GitHub repos     — README + recursive file tree + repo metadata
        • GitHub files     — raw source (blob / tree / raw / raw.githubusercontent.com)
        • PDFs             — full text (needs pypdf, pdfminer.six, or pdftotext CLI)
        • arXiv papers     — abstract + full text extracted from PDF
        • HTML pages       — content extraction (strips nav / ads / scripts)
        • Documentation    — readthedocs, GitHub Pages, official docs
        • Plain text / JSON / XML — stored as-is
        • REST API URLs    — raw response stored for reference

        Large documents are chunked automatically so every section is
        independently searchable. Content is visible to all agents instantly.

        Args:
            url:         Full URL to fetch (http or https).
            description: Why this resource matters — stored as context alongside the content.
        """
        try:
            source_type, content = _fetch_url_content(url, cfg)
        except urllib.error.HTTPError as exc:
            return f"HTTP {exc.code} {exc.reason} — {url}"
        except urllib.error.URLError as exc:
            return f"Network error ({exc.reason}) — {url}"
        except Exception as exc:
            return f"Fetch error: {exc} — {url}"

        content = content.strip()
        if not content:
            return f"Fetched {url} but found no readable content (type: {source_type})."

        label   = description.strip() or url
        mem_ids = _store_resource(
            store, cfg,
            url=url, content=content, label=label,
            agent_name=agent_name, source_type=source_type,
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

        Use for REST APIs, webhooks, or any endpoint requiring custom headers
        or a request body. Optionally stores the response in shared memory.

        Args:
            method:         HTTP verb — GET, POST, PUT, PATCH, DELETE, HEAD.
            url:            Full URL including query string.
            headers_json:   JSON object of extra request headers (default '{}').
                            E.g. '{"Authorization": "Bearer TOKEN", "Content-Type": "application/json"}'
            body:           Request body string (for POST / PUT).
            store_response: If True, saves the response to shared RAG memory.
            description:    Label used when store_response is True.
        """
        try:
            extra = json.loads(headers_json) if headers_json.strip() else {}
        except json.JSONDecodeError as exc:
            return f"headers_json parse error: {exc}"

        body_bytes = body.encode("utf-8") if body else None
        try:
            status, resp_hdrs, resp_bytes = _http_raw(
                url, cfg, method=method, extra_headers=extra, body=body_bytes
            )
        except urllib.error.URLError as exc:
            return f"Network error: {exc.reason}"
        except Exception as exc:
            return f"Request error: {exc}"

        ct = resp_hdrs.get("Content-Type", "unknown")
        try:
            resp_text = resp_bytes.decode("utf-8", errors="replace")
        except Exception:
            resp_text = f"[binary: {len(resp_bytes)} bytes]"

        lines = [
            f"HTTP {status}  {method.upper()} {url}",
            f"Content-Type: {ct}",
            f"Body length: {len(resp_bytes)} bytes",
            "---",
            resp_text[:4000],
        ]
        if len(resp_text) > 4000:
            lines.append(f"… [{len(resp_text) - 4000} chars truncated]")

        if store_response and resp_text.strip():
            label = description.strip() or f"{method.upper()} {url}"
            _store_resource(
                store, cfg,
                url=url, content=resp_text, label=label,
                agent_name=agent_name, source_type="api",
            )
            lines.append(f"\n[Stored in shared memory as: '{label}']")

        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────────
    @tool
    def search_web_resources(query: str, limit: int = 5, source_type: str = "") -> str:
        """Semantic search over all web content ingested by any agent.

        Always check this BEFORE fetching a new URL — the content may already
        be in shared memory from another agent. All agents share one store.

        Args:
            query:       What you're looking for (natural language).
            limit:       Max chunks to return (default 5).
            source_type: Optional filter — 'pdf', 'github', 'arxiv', 'html', 'api', 'raw'.
                         Leave empty to search all types.
        """
        try:
            candidates = store.search_memories(f"WEB_RESOURCE {query}", limit=limit * 6)
            hits = [r for r in candidates if "web_resource" in (r.get("tags") or "")]
            if source_type:
                hits = [r for r in hits if f"source_type:{source_type}" in (r.get("tags") or "")]
            hits = hits[:limit]

            if not hits:
                suffix = f" with source_type='{source_type}'" if source_type else ""
                return (
                    f"No web resources found{suffix} matching: {query!r}\n"
                    "Tip: use web_search → fetch_and_store_url to ingest content first."
                )

            lines = [f"Web resources matching {query!r} ({len(hits)} chunks):\n"]
            for r in hits:
                ts      = time.strftime("%Y-%m-%d", time.localtime(r.get("created_at", 0)))
                content = r.get("content", "")
                lines.append(f"[{ts}]")
                lines.append(content[:450])
                lines.append("---")
            return "\n".join(lines)
        except Exception as exc:
            return f"[WebTools Error] {exc}"

    # ──────────────────────────────────────────────────────────────────────
    @tool
    def list_web_resources(limit: int = 40) -> str:
        """List every URL fetched and stored by any agent in this session.

        Check this before fetching to avoid duplicate work.

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
