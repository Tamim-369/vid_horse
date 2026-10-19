"""Text helpers."""

from __future__ import annotations

import re

_NON_ALNUM = re.compile(r"[^a-z0-9]+")

_HTML_TAGS = re.compile(r"<[^>]*>")
_HTML_ENTITIES = (
    ("&nbsp;", " "),
    ("&amp;", "&"),
    ("&lt;", "<"),
    ("&gt;", ">"),
    ("&quot;", '"'),
    ("&#39;", "'"),
)

_TRAILING_SECTIONS = (
    "\nreferences",
    "\nexternal links",
    "\nsee also",
    "\nfurther reading",
    "\nbibliography",
    "\nnotes",
    "\nsources",
)


def strip_html(text: str) -> str:
    """Strip tags and common entities from an HTML snippet (e.g. captions)."""

    text = _HTML_TAGS.sub(" ", text)
    for entity, char in _HTML_ENTITIES:
        text = text.replace(entity, char)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def slugify(text: str) -> str:
    """Turn a title into a stable, URL-safe key.

    ``"Moon landing conspiracy theories"`` -> ``"moon-landing-conspiracy-theories"``.
    """

    return _NON_ALNUM.sub("-", text.strip().lower()).strip("-")


def clean_article_text(text: str) -> str:
    """Normalize wiki plain text and drop trailing apparatus sections.

    Reference lists, "See also" and similar end-matter add tokens without
    helping a small model understand the topic.
    """

    text = text.replace("\r", "")
    lowered = text.lower()

    cutoff = len(text)
    for marker in _TRAILING_SECTIONS:
        index = lowered.find(marker)
        if index != -1:
            cutoff = min(cutoff, index)
    text = text[:cutoff]

    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
