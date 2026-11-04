"""The chooser editor: one LLM pick per scene from the scored pool."""

from __future__ import annotations

import logging

from pydantic import BaseModel

logger = logging.getLogger(__name__)


CHOOSER_SYSTEM = """You are the image editor who makes the final pick for
one narration moment of a video. One image is selected from the candidates
by reading the file titles and descriptions alone.

Pick the candidate whose image gives the RIGHT VIBE for this narration
moment. The query describes a mood/atmosphere, not a literal subject.
Reject images that are clearly wrong for the mood.

REJECT (empty file) when:
- the image is clearly unrelated to the mood/vibe of the query
- the image is mostly TEXT — poster, cover, infographic, diagram, chart,
  comic, screenshot, map, scanned page, award banner
- it is a grave, tomb or cemetery shot when the subject is a living
  contribution, or any image that would look silly or tone-breaking on
  a serious narration channel (gags, parody, tongue-out poses)
- the image is a random person that doesn't match the mood

When several qualify, prefer the one that best matches the mood/atmosphere
described in the query, in a photograph or faithful artistic depiction,
with no text overlay.
"""

class PickSchema(BaseModel):
    """The chooser's verdict: one file title, or "" to reject all."""

    file: str = ""


class ChooseMixin:
    """Mixin: the final per-scene pick, attached to AssetResearchBlock."""

    def _choose(
        self,
        scene: str,
        search_words: str,
        candidates: list[tuple[int, dict]],
        confirm_only: bool = False,
    ) -> dict | None:
        """One editor-pick decision over a ranked shortlist.

        The chooser reads each candidate's own metadata like a human editor:
        for a person query the description must identify the right person;
        anything that cannot be verified, or a mostly-text image, is
        rejected outright. With no LLM wired in the top-ranked candidate
        passes unchanged.

        With ``confirm_only`` (CLIP already ranked by depicted content) the
        LLM may only veto the top candidate — it cannot down-shift to a
        pixel-weaker pick the way it did when it chose by caption alone.
        """

        if not candidates:
            return None
        if self._llm is None:
            return candidates[0][1]
        confirm = confirm_only
        lines = "\n".join(
            f"[{index}] file: {info.get('file', '')}\n"
            f"    description: {(info.get('description') or '').strip()[:260]}"
            for index, (_score, info) in enumerate(candidates)
        )
        if confirm:
            instruction = (
                "The FIRST candidate is the current pick. Study its own "
                "description. Reply its exact file title unchanged unless it "
                "breaks a veto rule: wrong-person or unverifiable identity, "
                "era mismatch, gag/parody pose, mostly-text, or visually "
                "irrelevant to the subject. To veto, reply file = empty string."
            )
        else:
            instruction = (
                "Pick the one candidate the viewer should see. "
                "file = its exact file title, or empty string to reject all."
            )
        messages = [
            {
                "role": "system",
                "content": CHOOSER_SYSTEM,
            },
            {
                "role": "user",
                "content": (
                    f"NARRATION (this moment): {scene}\n"
                    f"SUBJECT QUERY: {search_words}\n"
                    f"CANDIDATES:\n{lines}\n"
                    f"{instruction}"
                ),
            },
        ]
        try:
            verdict = self._llm.complete_json(messages, PickSchema)
        except Exception as exc:
            logger.warning("Chooser call failed: %s", exc)
            return candidates[0][1]
        wanted = (verdict.file or "").strip()
        if not wanted:
            logger.info("Chooser rejected all candidates for '%s'", search_words)
            return None
        if confirm:
            # veto-only: the pixel ranking stays the master
            return candidates[0][1]
        for _score, info in candidates:
            if info.get("file", "").lower() == wanted.lower():
                return info
        return None
