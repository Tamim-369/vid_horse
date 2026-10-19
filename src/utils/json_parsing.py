"""Helpers for coaxing structured data out of free-form LLM output."""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, ValidationError

from core.errors import LLMError
from core.interfaces import LLMMessage

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(text: str) -> str:
    """Best-effort extraction of a JSON object from ``text``.

    Strips markdown code fences and surrounding prose by slicing from the first
    ``{`` to the last ``}``.
    """

    match = _FENCE_RE.search(text)
    candidate = match.group(1) if match else text

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end < start:
        return candidate.strip()
    return candidate[start : end + 1]


def parse_json_as[TModel: BaseModel](text: str, schema: type[TModel]) -> TModel:
    """Parse ``text`` into a ``schema`` instance or raise :class:`LLMError`."""

    raw = extract_json(text)
    try:
        return schema.model_validate_json(raw)
    except ValidationError:
        try:
            return schema.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise LLMError(
                f"Could not parse LLM response as {schema.__name__}: {exc}"
            ) from exc


def json_schema_instruction(schema: type[BaseModel]) -> LLMMessage:
    """Message instructing the model to emit JSON matching ``schema``."""

    return {
        "role": "system",
        "content": (
            "Respond with a single, valid JSON object that conforms to the "
            "following JSON Schema. Do not include markdown fences or any "
            "commentary.\n\n"
            f"{json.dumps(schema.model_json_schema(), indent=2)}"
        ),
    }


def repair_instruction(error: Exception) -> str:
    """Message asking the model to fix a schema-invalid response."""

    return (
        "Your previous response was not valid against the required schema. "
        f"Validation error: {error}\n"
        "Return ONLY a corrected JSON object. No commentary, no markdown fences."
    )
