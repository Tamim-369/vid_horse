#!/usr/bin/env python3
"""Probe local Ollama models for structured-output reliability and speed.

Runs the same JSON-schema-constrained task against several models and reports
how often the output is schema-valid plus how fast each model generates. Used
to choose the default model for the content pipeline on CPU-only hardware.

Usage:
    uv run python experiments/probe_models.py
    uv run python experiments/probe_models.py --models gemma3:4b,gemma3:1b --runs 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import urllib.error
import urllib.request

DEFAULT_MODELS = [
    "qwen2.5:3b",
    "qwen3:4b",
    "phi4-mini:latest",
    "gemma3:4b",
    "llama3.2:latest",
]

SCHEMA = {
    "type": "object",
    "properties": {
        "theory": {"type": "string"},
        "emotional_appeal": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 4,
        },
        "cognitive_mechanisms": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 4,
        },
        "why_taken_seriously": {"type": "string"},
    },
    "required": [
        "theory",
        "emotional_appeal",
        "cognitive_mechanisms",
        "why_taken_seriously",
    ],
}

SYSTEM_PROMPT = (
    "You are a cultural analyst who explains conspiracy theories without "
    "asserting whether they are true. Focus on why the belief is compelling "
    "and why some people took it seriously. Answer with JSON only."
)

USER_PROMPT = (
    'Analyse the conspiracy theory: "The Moon landings were faked."\n'
    "Return a JSON object with the theory name, 2-4 emotional appeals, "
    "2-4 cognitive mechanisms, and one sentence on why people took it "
    "seriously."
)


def _post(host: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        f"{host}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def chat(
    host: str,
    model: str,
    timeout: float,
    num_ctx: int,
    num_predict: int,
    seed: int,
) -> tuple[dict, dict]:
    """Call Ollama once; retry without the ``think`` field if unsupported."""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        "stream": False,
        "format": SCHEMA,
        "options": {
            "temperature": 0,
            "seed": seed,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
        "keep_alive": "2m",
        "think": False,
    }
    try:
        return _post(host, payload, timeout), payload["options"]
    except urllib.error.HTTPError:
        payload.pop("think", None)
        return _post(host, payload, timeout), payload["options"]


def validate(output: str) -> tuple[bool, list[str]]:
    """Minimal validator for the probe schema (no external deps)."""

    errors: list[str] = []
    try:
        data = json.loads(output)
    except json.JSONDecodeError as exc:
        return False, [f"invalid JSON: {exc}"]

    if not isinstance(data, dict):
        return False, ["top level is not an object"]

    for key in SCHEMA["required"]:
        if key not in data:
            errors.append(f"missing '{key}'")

    for key, spec in SCHEMA["properties"].items():
        value = data.get(key)
        if spec["type"] == "string":
            if not isinstance(value, str) or not value.strip():
                errors.append(f"'{key}' is not a non-empty string")
        elif spec["type"] == "array":
            if not isinstance(value, list):
                errors.append(f"'{key}' is not a list")
                continue
            if not all(isinstance(item, str) and item.strip() for item in value):
                errors.append(f"'{key}' has non-string items")
            elif not (spec["minItems"] <= len(value) <= spec["maxItems"]):
                errors.append(
                    f"'{key}' has {len(value)} items (want "
                    f"{spec['minItems']}-{spec['maxItems']})"
                )
    return not errors, errors


def _tokens_per_second(response: dict) -> float:
    eval_count = response.get("eval_count") or 0
    eval_duration = response.get("eval_duration") or 0
    if not eval_count or not eval_duration:
        return 0.0
    return eval_count / (eval_duration / 1e9)


def probe_model(
    host: str,
    model: str,
    runs: int,
    timeout: float,
    num_ctx: int,
    num_predict: int,
) -> dict:
    valid = 0
    speeds: list[float] = []
    latencies: list[float] = []
    tokens: list[int] = []
    sample = ""
    notes: list[str] = []

    for run in range(runs):
        try:
            response, _ = chat(host, model, timeout, num_ctx, num_predict, seed=42 + run)
        except (urllib.error.URLError, TimeoutError) as exc:
            notes.append(f"run {run + 1}: request failed: {exc}")
            break

        if "error" in response:
            notes.append(f"run {run + 1}: {response['error']}")
            break

        content = response.get("message", {}).get("content", "")
        if run == 0:
            sample = content
        ok, errors = validate(content)
        if ok:
            valid += 1
        else:
            notes.append(f"run {run + 1}: " + "; ".join(errors))

        speeds.append(_tokens_per_second(response))
        tokens.append(response.get("eval_count") or 0)
        total = response.get("total_duration") or 0
        latencies.append(total / 1e9)

    completed = len(latencies)
    return {
        "model": model,
        "valid": valid,
        "runs": completed,
        "tok_s": statistics.mean(speeds) if speeds else 0.0,
        "latency": statistics.mean(latencies) if latencies else 0.0,
        "tokens": round(statistics.mean(tokens)) if tokens else 0,
        "sample": sample,
        "notes": notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--num-predict", type=int, default=400)
    parser.add_argument("--show-samples", action="store_true")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    print(f"Probing {len(models)} model(s), {args.runs} run(s) each on {args.host}\n")
    print(f"{'model':<20} {'valid':>7} {'tok/s':>8} {'latency':>9} {'out_tok':>8}")
    print("-" * 56)

    results = []
    for model in models:
        result = probe_model(
            args.host, model, args.runs, args.timeout, args.num_ctx, args.num_predict
        )
        results.append(result)
        print(
            f"{result['model']:<20} "
            f"{result['valid']}/{result['runs']:<5} "
            f"{result['tok_s']:>8.1f} "
            f"{result['latency']:>8.1f}s "
            f"{result['tokens']:>8}"
        )
        for note in result["notes"]:
            print(f"    - {note}")
        if args.show_samples and result["sample"]:
            print("    sample:", result["sample"][:300].replace("\n", " "))
        print()


if __name__ == "__main__":
    main()
