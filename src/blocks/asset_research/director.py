"""The two asset director agents: visual intent, then source-routed queries.

Layer 1 (:class:`VisualDirectorBlock`) reads the script and says what the
viewer should see for every scene — one concrete subject per distinct topic
the narration touches, each with the narration phrase that triggers it.

Layer 2 (:class:`QueryBuilderBlock`) turns each subject into search keywords
and routes them: real space imagery to the NASA image archive, everything
else to topical free sources (Commons first, Openverse explicit only).
"""

from __future__ import annotations

import json

from core.interfaces import LLMProvider
from core.models import (
    IntentPack,
    QueryPack,
    ResearchBrief,
    VideoScript,
)

_TEMPERATURE = 0.4

DIRECTOR_SYSTEM = """You are the visual director of a conspiracy-theory video channel.

You get a JSON list of scenes (narration + visual line) from a video script.
For every scene you pick what the viewer should SEE while it plays — one
image PER distinct subject the narration touches. If the narration first
talks about pyramids and then about stars, that scene needs exactly two
subjects: pyramids and stars.

THE RULES
1. Every subject must be grounded in what the scene actually talks about.
   If it is not about this theory, you have failed.
2. Subjects are concrete and specific: a place, an object, a document, a
   type of person visibly doing something. Name proper nouns from the
   narration where they exist (Chichén Itzá, the EC-47Q, Lavrentiy Beria).
3. A generic decorative image ("dreamy sky", "mysterious fog") is FORBIDDEN.
   Space/planet/galaxy/black-hole imagery is only generic if the scene
   literally discusses space or the cosmos — never as mood filler.
4. FORBIDDEN subjects: anything that displays mostly text — posters,
   newspaper front pages, documents, film title cards, screenshots.
   Also NO camera/composition subjects (close-up, aerial view) and
   NO mood or colour words (dark, moody, dramatic). Subject content only.
5. Each subject: `intent` is at most 9 words, English, no punctuation.
6. `trigger` is a short phrase of AT MOST SIX consecutive narration words
   where that subject comes up in the speech. Copy the words verbatim from
   the narration (closest 2-3 words when none match cleanly). The trigger
   decides WHEN the image appears; order the subjects by trigger position.
7. One subject per distinct narration topic; 2 subjects when the narration
   clearly moves to a second topic, 3 at most.

"""


HORROR_DIRECTOR_NOTES = """
THIS IS A HORROR STORY (fiction), not a documentary.

1. Subjects are the story's SETTING and PROPS: real, everyday, photographable
   things — a dim apartment corridor, an old light switch, a bathroom mirror,
   a basement door, a worn family photo, a night street. Choose what the
   beat HAPPENS ON, never the supernatural event itself.
2. FORBIDDEN subjects: anything describing the entity or a ghostly effect
   (whispers, silhouettes, breath, shadows moving by themselves) — those
   cannot be photographed. Show the mundane place/object instead.
3. Mood words (dark, eerie, moody) are allowed as ONE keyword inside intent
   phrasing when they describe an actual place ("empty corridor at night").
4. Never use the story title as a subject. Grounded in this beat only.
"""

HORROR_QUERY_NOTES = """
THIS IS A HORROR STORY (fiction). Search for the SETTING and PROPS only:
- 1-3 catalogue words naming the real place or object ("apartment hallway",
  "bathroom mirror", "basement door", "family photograph"). One atmosphere
  word allowed when it names the real subject ("corridor night").
- NEVER search the story title or narration words like whisper, shadow,
  silhouette, ghost, entity, fear.
- Still prefer "commons"; nasa stays wrong unless the story literally
  enters space.
"""

QUERY_SYSTEM = """You route image searches for a video channel.

You get JSON of scenes; each scene has subjects to show. For every subject
produce ONE search. The query must name the THING the viewer should see —
never what people did, only what the image should depict.

HARD RULES (violating any of these is a failure):
1. The query names the object, place, structure or artefact the subject
   describes — the invention itself, not a group of people. If the
   narration says "what the team built", the query is the BUILDING, the
   MACHINE or the OBJECT, never "scientists", "researchers", "team",
   "meeting" or any group-of-people word.
2. A person's bare proper name (e.g. "einstein") is allowed ONLY when the
   subject itself is that person. Otherwise find the thing they made.
3. No narration verbs, adverbs or story words ("discovered", "proved",
   "imagine"). Camera words ("close-up", "aerial") are banned.
4. 1-3 real catalogue keywords, lowercase, no punctuation, no quotation
   marks. GOOD: "maya glyphs", "interferometer", "voyager spacecraft".
   BAD: "scientists discovering", "crowd waiting in fear", full sentences.
5. One query per subject — subjects come from distinct sentence topics, so
   each query belongs to exactly one narration moment. Copy the subject's
   `trigger` through unchanged.

SOURCE RULES
- "nasa" when the subject is literally about space itself: planets, the
  moon, the sun, stars, black holes, galaxies, nebulae, comets, asteroids,
  orbits, space missions and their craft. NASA's archive is
  public-domain and photography-first.
- "commons" for people, portraits, historic documents, buildings,
  instruments on Earth, ruins, war, everyday places.
  GOOD commons queries: "einstein", "interferometer", "soviet radio".
  GOOD nasa queries: "cygnus x-1", "black hole", "milky way".

Copy each subject's `trigger` through unchanged.

"""


class IntentSchema(IntentPack):
    """Top-level JSON the director agent returns."""


class RouterSchema(QueryPack):
    """Top-level JSON the routing agent returns."""


def _is_fiction(script: VideoScript) -> bool:
    return any("original horror story" in note.lower() for note in script.source_notes)


class VisualDirectorBlock:
    """Agent 1: script narration -> per-scene visual subjects."""

    name = "visual_direction"

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    def run(self, theory, script: VideoScript) -> IntentPack:
        payload = [
            {
                "scene_index": -1,
                "visual": "(hook line)",
                "narration": script.hook,
            },
            *[
                {
                    "scene_index": index,
                    "visual": scene.visual,
                    "narration": scene.narration,
                }
                for index, scene in enumerate(script.scenes)
            ],
        ]
        system = DIRECTOR_SYSTEM + (HORROR_DIRECTOR_NOTES if _is_fiction(script) else "")
        answer = self._llm.complete_json(
            [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": (
                        f"Theory: {theory.name}\nHook of the video: {script.hook}\n"
                        f"Scenes: {json.dumps(payload, ensure_ascii=False)}"
                    ),
                },
            ],
            IntentSchema,
            temperature=_TEMPERATURE,
        )
        return IntentPack(
            slug=theory.slug,
            topic=theory.name,
            scenes=answer.scenes,
            fiction=_is_fiction(script),
        )


class QueryBuilderBlock:
    """Agent 2: subjects -> keywords + source routing (commons/nasa/openverse)."""

    name = "query_routing"

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    def run(self, brief: ResearchBrief, intents: IntentPack) -> QueryPack:
        payload = [item.model_dump() for item in intents.scenes]
        system = QUERY_SYSTEM + (HORROR_QUERY_NOTES if intents.fiction else "")
        answer = self._llm.complete_json(
            [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": (
                        f"Topic: {brief.title}\nSummary: {brief.one_line}\n"
                        f"Scenes: {json.dumps(payload, ensure_ascii=False)}"
                    ),
                },
            ],
            RouterSchema,
            temperature=_TEMPERATURE,
        )
        return QueryPack(
            slug=intents.slug,
            topic=intents.topic,
            scenes=answer.scenes,
            fiction=intents.fiction,
        )
