"""Horror story corpus: pull CC BY-SA creepypastas from the Fandom wiki.

Fetched stories are the *study material* for our writer agents — pattern
mining, few-shot seeds — never content we publish. Every record keeps its
source URL and license tag.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import httpx

from config.settings import Settings

logger = logging.getLogger(__name__)

API = "https://creepypasta.fandom.com/api.php"
LICENSE = "CC BY-SA 3.0"
_MIN_CHARS = 600
_MAX_CHARS = 20000
_PAGE_SLEEP = 0.3

_MARKUP_FILE = re.compile(r"\[\[File:[^\]]*\]\]", re.IGNORECASE)
_MARKUP_LINK = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]")
_MARKUP_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
_MARKUP_HEADING = re.compile(r"^=+\s*(.+?)\s*=+$", re.MULTILINE)
_MARKUP_EMPHASIS = re.compile(r"('{2,4})(.*?)\1")
_MARKUP_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_MARKUP_TAG = re.compile(r"<[^>]+>")
_MARKUP_FILELINE = re.compile(r"^(?:File|Image):.*$", re.MULTILINE | re.IGNORECASE)
_MARKUP_BLANKS = re.compile(r"\n{3,}")


def _clean_wikitext(raw: str) -> str:
    text = _MARKUP_COMMENT.sub("", raw)
    text = _MARKUP_TEMPLATE.sub("", text)
    text = _MARKUP_FILE.sub("", text)
    text = _MARKUP_FILELINE.sub("", text)
    text = _MARKUP_LINK.sub(r"\1", text)
    text = _MARKUP_EMPHASIS.sub(r"\2", text)
    text = _MARKUP_HEADING.sub("", text)
    text = _MARKUP_TAG.sub("", text)
    text = _MARKUP_BLANKS.sub("\n\n", text)
    return text.strip()


def _page_titles(client: httpx.Client, count: int) -> list[str]:
    titles: list[str] = []
    cont: dict[str, str] | None = None
    while len(titles) < count:
        params: dict[str, Any] = {
            "action": "query",
            "list": "allpages",
            "aplimit": "max",
            "format": "json",
        }
        if cont:
            params.update(cont)
        response = client.get(API, params=params)
        response.raise_for_status()
        payload = response.json()
        titles.extend(page["title"] for page in payload["query"]["allpages"])
        cont = payload.get("continue")
        if not cont:
            break
    return titles[:count]


def _fetch_story(client: httpx.Client, title: str) -> dict[str, Any] | None:
    response = client.get(
        API,
        params={
            "action": "parse",
            "page": title,
            "prop": "wikitext",
            "redirects": "1",
            "format": "json",
        },
    )
    response.raise_for_status()
    payload = response.json()
    if "parse" not in payload:
        return None
    text = _clean_wikitext(payload["parse"]["wikitext"]["*"])
    if not _MIN_CHARS <= len(text) <= _MAX_CHARS:
        return None
    title = payload["parse"].get("title", title)
    slug = title.replace(" ", "_")
    return {
        "title": title,
        "url": f"https://creepypasta.fandom.com/wiki/{slug}",
        "text": text,
        "source": "creepypasta.fandom.com",
        "license": LICENSE,
    }


def pull_stories(settings: Settings, count: int, out_path: Path) -> Path:
    """Fetch ``count`` stories and append-dedupe them into a JSONL corpus."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    have: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                have.add(json.loads(line)["title"])
    client = httpx.Client(
        headers={"User-Agent": settings.http_user_agent},
        timeout=settings.http_timeout,
        follow_redirects=True,
    )
    titles = [t for t in _page_titles(client, count * 3) if t not in have]
    written = len(have)
    try:
        for title in titles:
            if written >= count:
                break
            story = _fetch_story(client, title)
            time.sleep(_PAGE_SLEEP)
            if not story or story["title"] in have:
                continue
            with out_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(story, ensure_ascii=False) + "\n")
            have.add(story["title"])
            written += 1
            if written % 25 == 0:
                logger.info("Corpus pull: %d/%d", written, count)
    finally:
        client.close()
    logger.info("Corpus ready: %d stories at %s", written, out_path)
    return out_path
