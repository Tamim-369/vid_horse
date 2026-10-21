"""Space-facts connector: a curated catalog of cosmic-scale topics.

The dread-of-scale niche lives on public-domain NASA/ESO imagery, so the
discovery step is simply a static catalog of cosmic objects and events —
each with its Wikipedia URL for the research block. No scraping.
"""

from __future__ import annotations

from collections.abc import Iterator
from urllib.parse import quote

from core.models import SourceTheory

name = "space"

_TOPICS: tuple[str, ...] = (
    "Black hole",
    "Supermassive black hole",
    "Sagittarius A*",
    "Neutron star",
    "Pulsar",
    "Magnetar",
    "Supernova",
    "White dwarf",
    "Red giant",
    "Quasar",
    "Rogue planet",
    "Brown dwarf",
    "The Milky Way",
    "Andromeda Galaxy",
    "Local Group",
    "Boötes Void",
    "Great Attractor",
    "Dark matter halo",
    "Cosmic microwave background",
    "Observable universe",
    "Big Bang",
    "Heat death of the universe",
    "Voyager 1",
    "Pioneer 10",
    "New Horizons",
    "Philae lander",
    "Oumuamua",
    "Halley's Comet",
    "Oort Cloud",
    "Kuiper Belt",
    "Asteroid belt",
    "The Sun",
    "Coronal mass ejection",
    "Solar eclipse",
    "The Moon",
    "Titan (moon)",
    "Europa (moon)",
    "Io (moon)",
    "Enceladus",
    "Saturn",
    "Jupiter",
    "Great Red Spot",
    "Mars",
    "Valles Marineris",
    "Olympus Mons",
    "Venus",
    "Betelgeuse",
    "Pillars of Creation",
    "Crab Nebula",
    "Cygnus X-1",
    "V838 Monocerotis",
    "Proxima Centauri",
    "TRAPPIST-1",
    "Kepler-452b",
    "Interstellar medium",
    "Cosmic dust",
    "Exoplanet",
    "Binary star",
    "Galaxy filament",
    "Supervoid",
)


class SpaceConnector:
    """Yield the static cosmic catalog as discovery hits."""

    name = name

    def __init__(self) -> None:
        pass

    def discover(
        self,
        *,
        limit: int | None = None,
        with_metrics: bool = False,  # noqa: ARG002 - interface parity
    ) -> Iterator[SourceTheory]:
        for topic in _TOPICS[: limit or len(_TOPICS)]:
            yield SourceTheory(
                source=self.name,
                name=topic,
                url=f"https://en.wikipedia.org/wiki/{quote(topic.replace(' ', '_'))}",
                categories=["space"],
            )
