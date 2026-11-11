"""Base class for pipeline blocks and a shared structured-retry helper.

Retry loops are strictly SERIAL: one LLM request in flight at any time.
Parallel "race waves" were removed after concurrent requests made the
ollama daemon hold several model slots at once and hard-crash a box.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence

from pydantic import BaseModel

from core.errors import BlockError, LLMError
from core.interfaces import LLMMessage, LLMProvider
from utils.json_parsing import parse_json_as

logger = logging.getLogger(__name__)


class Block:
    """A single composable stage of the pipeline."""

    name: str = "block"


def _one_attempt(
    llm: LLMProvider,
    conversation: list[LLMMessage],
    parse: Callable[[str], object],
    temperature: float,
) -> tuple[object | None, Exception | None, str]:
    """One generate→parse shot: (result, None, raw) or (None, error, raw)."""

    raw = ""
    try:
        raw = llm.complete(conversation, temperature=temperature)
    except LLMError as exc:
        return None, exc, raw
    try:
        return parse(raw), None, raw
    except BlockError as exc:
        # the rejected text was cached the moment it came back; drop it so
        # future runs sample fresh instead of re-reading poisoned output
        invalidate = getattr(llm, "invalidate", None)
        if invalidate is not None:
            try:
                invalidate(conversation, temperature)
            except Exception as cache_exc:
                logger.debug("Could not evict cached sample: %s", cache_exc)
        return None, exc, raw
    except Exception as exc:
        return None, BlockError(f"{type(exc).__name__}: {exc}"), raw


def generate_parsed[T](
    llm: LLMProvider,
    messages: Sequence[LLMMessage],
    parse: Callable[[str], T],
    *,
    attempts: int,
    temperature: float,
    corrective: Callable[[Exception], str],
    label: str = "output",
) -> T:
    """Generate text and parse it, retrying with corrective feedback on failure.

    ``parse`` should raise :class:`~core.errors.BlockError` when the text is
    unusable; generation failures (:class:`~core.errors.LLMError`) are also
    retried, since a fresh sample often succeeds. Attempts run one at a
    time — never concurrently.
    """

    conversation = list(messages)
    last_error: Exception | None = None
    raw = ""
    spent = 0
    total = max(1, attempts)

    while spent < total:
        started = time.monotonic()
        result, error, sample = _one_attempt(llm, conversation, parse, temperature)
        spent += 1
        if result is not None:
            logger.info(
                "%s: attempt accepted in %.0fs", label, time.monotonic() - started
            )
            return result
        last_error, raw = error, sample
        logger.warning(
            "%s: attempt %d/%d rejected (%s); sample started with %r",
            label,
            spent,
            total,
            error,
            sample[:160],
        )

        if spent >= total:
            break
        conversation = [
            *messages,
            {"role": "assistant", "content": raw},
            {"role": "user", "content": corrective(last_error)},
        ]

    raise BlockError(
        f"Could not produce valid {label} after {spent} attempts: {last_error}"
    )


def _one_json_attempt[TModel: BaseModel](
    llm: LLMProvider,
    messages: Sequence[LLMMessage],
    schema: type[TModel],
    temperature: float,
    quality: Callable[[TModel], list[str]] | None,
) -> tuple[TModel | None, Exception | None, str]:
    """One structured generate→parse→quality shot; errors carry the sample."""

    raw = ""
    try:
        raw = llm.complete_structured(messages, schema, temperature=temperature)
        # normalize smart quotes to straight quotes so JSON parsing survives
        raw = (
            raw.replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u2018", "'")
            .replace("\u2019", "'")
        )
        answer = parse_json_as(raw, schema)
    except LLMError as exc:
        return None, exc, raw
    except Exception as exc:
        return None, BlockError(f"{type(exc).__name__}: {exc}"), raw
    if quality is not None:
        problems = quality(answer)
        if problems:
            # the poisoned sample was cached; evict so the next attempt
            # samples fresh instead of re-reading it
            invalidate = getattr(llm, "invalidate", None)
            if invalidate is not None:
                try:
                    invalidate(messages, temperature, schema)
                except Exception as cache_exc:
                    logger.debug("Could not evict cached JSON: %s", cache_exc)
            return None, BlockError("; ".join(problems)), raw
    return answer, None, raw


def generate_json_parsed[TModel: BaseModel](
    llm: LLMProvider,
    messages: Sequence[LLMMessage],
    schema: type[TModel],
    *,
    attempts: int,
    temperature: float,
    quality: Callable[[TModel], list[str]] | None = None,
    label: str = "json",
) -> TModel:
    """Generate structured JSON, retrying with corrective feedback on failure.

    Same shape as :func:`generate_parsed` but on ``complete_structured``.
    Attempts run one at a time — never concurrently.
    """

    corrective = _json_corrective
    last_error: Exception | None = None
    total = max(1, attempts)
    for attempt in range(total):
        started = time.monotonic()
        result, error, raw = _one_json_attempt(
            llm, messages, schema, temperature, quality
        )
        if result is not None:
            logger.info("%s: attempt accepted in %.0fs", label, time.monotonic() - started)
            return result
        last_error, raw_failed = error, raw
        logger.warning(
            "%s: attempt %d/%d rejected (%s); sample started with %r",
            label, attempt + 1, total, error, raw_failed[:160],
        )
        if attempt + 1 < total:
            messages = [
                *messages,
                {"role": "assistant", "content": raw_failed},
                {"role": "user", "content": corrective(last_error)},
            ]
    raise BlockError(f"Could not produce valid {label} after {total} attempts: {last_error}")


def _json_corrective(error: Exception) -> str:
    """One-shot corrective instruction for structured JSON stages."""

    return (
        "Your previous answer failed validation:\n"
        f"{error}\n\n"
        "Produce the COMPLETE corrected answer as strict JSON only -"
        " no prose, no markdown, no code fences, follow every field rule."
    )
