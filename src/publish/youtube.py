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
            raise ProviderError(f"YouTube token not found: {self._token_path} — run 'nebula youtube-auth' once")
        data = json.loads(self._token_path.read_text(encoding="utf-8"))
        creds = Credentials.from_authorized_user_info(data)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            self._token_path.write_text(creds.to_json(), encoding="utf-8")
        if not creds.valid:
            raise ProviderError("YouTube credentials invalid; re-run 'nebula youtube-auth'")
        return creds

    def authenticate(self) -> None:
        """One-time browser consent: exchange client secrets for a stored token.

        Requires an OAuth client of type DESKTOP (the downloaded
        client_secret JSON) and the user clicking through the consent
        screen once. The refresh token is stored locally and refreshed
        automatically afterwards.
        """

        from google_auth_oauthlib.flow import InstalledAppFlow

        if not self._secrets_path.exists():
            raise ProviderError(
                f"Client secrets not found: {self._secrets_path} — set YOUTUBE_CLIENT_SECRETS in .env"
            )
        scopes = [
            "https://www.googleapis.com/auth/youtube.upload",
            "https://www.googleapis.com/auth/youtube.readonly",
        ]
        flow = InstalledAppFlow.from_client_secrets_file(str(self._secrets_path), scopes)
        creds = flow.run_local_server(
            port=self._settings.youtube_redirect_port,
            access_type="offline",
            prompt="consent",  # force a refresh token on the first grant
        )
        self._token_path.parent.mkdir(parents=True, exist_ok=True)
        self._token_path.write_text(creds.to_json(), encoding="utf-8")
        logger.info("YouTube token saved to %s", self._token_path)

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
        # a flaky connection can abort a resumable upload mid-chunk; retrying
        # the insert is quota-safe (an aborted session never creates a video)
        last_exc: Exception | None = None
        for attempt in range(3):
            media = MediaFileUpload(str(video_path), chunksize=-1, resumable=True)
            request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
            response = None
            try:
                while response is None:
                    status, response = request.next_chunk()
                    if status:
                        logger.info("Upload %d%%", int(status.progress() * 100))
                break
            except Exception as exc:  # noqa: BLE001 - network/5xx failures
                last_exc = exc
                wait = 5.0 * (attempt + 1)
                logger.warning("upload attempt %d failed (%s); retrying in %.0fs", attempt + 1, exc, wait)
                if attempt < 2:
                    import time

                    time.sleep(wait)
        else:
            raise ProviderError(f"YouTube upload failed after 3 attempts: {last_exc}") from last_exc
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
