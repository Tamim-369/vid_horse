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


def extract_json_value(text: str) -> object:
    """First complete JSON value in ``text`` (object or array), prose ignored.

    Small local models happily wrap answers in commentary or return a bare
    array where the schema wanted an object; :func:`extract_json` only
    handles ``{...}`` so it shreds top-level arrays. This walks the text and
    lets the JSON decoder tell us where a real value starts and ends.
    """

    match = _FENCE_RE.search(text)
    candidate = match.group(1) if match else text
    decoder = json.JSONDecoder()
    for index, char in enumerate(candidate):
        if char not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(candidate[index:])
            return value
        except json.JSONDecodeError:
            continue
    raise LLMError("no JSON value found in LLM response")


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
