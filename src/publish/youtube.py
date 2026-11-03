"""YouTube Data API v3 publishing helper."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from blocks.title_description.schema import TitlePack
from config.settings import get_settings
from core.errors import ProviderError

logger = logging.getLogger(__name__)


class YouTubePublisher:
    def __init__(
        self,
        client_secrets_path: Path | None = None,
        token_path: Path | None = None,
    ) -> None:
        settings = get_settings()
        self._secrets_path = Path(client_secrets_path or settings.youtube_client_secrets_path)
        self._token_path = Path(token_path or settings.youtube_token_path)
        self._settings = settings

    def _credentials(self):
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials

        if not self._token_path.exists():
            raise ProviderError(f"YouTube token not found: {self._token_path}")
        data = json.loads(self._token_path.read_text(encoding="utf-8"))
        creds = Credentials.from_authorized_user_info(data)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            self._token_path.write_text(creds.to_json(), encoding="utf-8")
        if not creds.valid:
            raise ProviderError("YouTube credentials invalid; re-run OAuth flow")
        return creds

    def upload(
        self,
        video_path: Path,
        title_pack: TitlePack,
        *,
        thumbnail_path: Path | None = None,
    ) -> str:
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload

        creds = self._credentials()
        youtube = build("youtube", "v3", credentials=creds)
        body = {
            "snippet": {
                "title": title_pack.title[:100],
                "description": title_pack.description[:5000],
                "tags": title_pack.tags[:500],
                "categoryId": self._settings.youtube_category_id,
            },
            "status": {
                "privacyStatus": self._settings.youtube_privacy_status,
                "selfDeclaredMadeForKids": False,
            },
        }
        media = MediaFileUpload(str(video_path), chunksize=-1, resumable=True)
        request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                logger.info("Upload %d%%", int(status.progress() * 100))
        video_id = response["id"]
        logger.info("Uploaded video id=%s", video_id)

        if thumbnail_path is not None and thumbnail_path.exists():
            try:
                youtube.thumbnails().set(
                    videoId=video_id, media_body=MediaFileUpload(str(thumbnail_path))
                ).execute()
            except Exception as exc:  # thumbnails require verified channel
                logger.warning("Thumbnail upload skipped: %s", exc)

        return video_id

    def publish_if_enabled(self, video_path: Path, title_pack: TitlePack) -> str | None:
        if not self._settings.auto_publish:
            logger.info("AUTO_PUBLISH disabled; skipping upload")
            return None
        return self.upload(video_path, title_pack)
