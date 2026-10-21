"""Discovery source connectors."""

from providers.sources.article import MediaWikiArticleFetcher
from providers.sources.base import SourceConnector, build_http_client
from providers.sources.fandom import FandomConnector
from providers.sources.mediawiki import MediaWikiClient
from providers.sources.nasa import NasaImagesClient
from providers.sources.openverse import OpenverseClient
from providers.sources.wikidata import WikidataConnector
from providers.sources.wikipedia import WikipediaConnector

__all__ = [
    "FandomConnector",
    "MediaWikiArticleFetcher",
    "MediaWikiClient",
    "NasaImagesClient",
    "OpenverseClient",
    "SourceConnector",
    "WikidataConnector",
    "WikipediaConnector",
    "build_http_client",
]
