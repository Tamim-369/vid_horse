"""Base class for pipeline blocks and a shared structured-retry helper."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

from core.errors import BlockError, LLMError
from core.interfaces import LLMMessage, LLMProvider

logger = logging.getLogger(__name__)


class Block:
    """A single composable stage of the pipeline."""

    name: str = "block"


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
    retried, since a fresh sample often succeeds.
    """

    conversation = list(messages)
    last_error: Exception | None = None

    for attempt in range(max(1, attempts)):
        try:
            raw = llm.complete(conversation, temperature=temperature)
        except LLMError as exc:
            last_error = exc
            logger.debug(
                "Generation failed for %s (attempt %d/%d): %s",
                label,
                attempt + 1,
                attempts,
                exc,
            )
            continue

        try:
            return parse(raw)
        except BlockError as exc:
            last_error = exc
            # the rejected text was cached the moment it came back; drop
            # it so the next attempt (and every future run) samples fresh
            # instead of re-reading the same poisoned response forever
            invalidate = getattr(llm, "invalidate", None)
            if invalidate is not None:
                try:
                    invalidate(conversation, temperature)
                except Exception as cache_exc:
                    logger.debug("Could not evict cached sample: %s", cache_exc)
            logger.debug(
                "Repairing %s (attempt %d/%d): %s",
                label,
                attempt + 1,
                attempts,
                exc,
            )
            conversation = [
                *messages,
                {"role": "assistant", "content": raw},
                {"role": "user", "content": corrective(exc)},
            ]

    raise BlockError(f"Could not produce valid {label} after {attempts} attempts: {last_error}")
