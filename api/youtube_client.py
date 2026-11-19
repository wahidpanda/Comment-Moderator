"""YouTube Data API v3 commentThreads.list with pagination."""

from __future__ import annotations

import os
import re
from typing import Any

import httpx

from engine.schemas import CommentInput

_YT_ID_PATTERNS = (
    re.compile(r"[?&]v=([A-Za-z0-9_-]{11})\b"),
    re.compile(r"youtube\.com/shorts/([A-Za-z0-9_-]{11})\b"),
    re.compile(r"youtube\.com/embed/([A-Za-z0-9_-]{11})\b"),
    re.compile(r"youtube\.com/live/([A-Za-z0-9_-]{11})\b"),
    re.compile(r"youtu\.be/([A-Za-z0-9_-]{11})\b"),
    re.compile(r"youtube\.com/watch\?v=([A-Za-z0-9_-]{11})\b"),
)
_BARE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_API = "https://www.googleapis.com/youtube/v3/commentThreads"
_VIDEOS_API = "https://www.googleapis.com/youtube/v3/videos"


class YouTubeAPIError(Exception):
    pass


def normalize_author_channel_id(raw: Any) -> str | None:
    """YouTube may return authorChannelId as a string or {"value": "UC…"}."""
    if raw is None:
        return None
    if isinstance(raw, str):
        s = raw.strip()
        return s or None
    if isinstance(raw, dict):
        val = raw.get("value")
        if val is None:
            return None
        return str(val).strip() or None
    return str(raw).strip() or None


def extract_video_id(url: str) -> str:
    text = url.strip().strip("\"'<>")
    if not text:
        raise YouTubeAPIError("Invalid YouTube URL")
    if _BARE_ID_RE.fullmatch(text):
        return text
    for pat in _YT_ID_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1)
    raise YouTubeAPIError("Invalid YouTube URL")


def _summarize_description(title: str, description: str, *, max_chars: int = 480) -> str:
    """Short video blurb from title + description (no LLM)."""
    body = " ".join(description.split())
    if not body:
        return title
    parts: list[str] = []
    for chunk in body.replace("!", ".").replace("?", ".").split("."):
        line = chunk.strip()
        if len(line) < 12:
            continue
        parts.append(line)
        if sum(len(p) for p in parts) >= max_chars:
            break
    blurb = ". ".join(parts[:4])
    if blurb and not blurb.endswith("."):
        blurb += "."
    if len(blurb) < 40:
        blurb = body[:max_chars] if body else title
    if len(blurb) > max_chars:
        blurb = blurb[: max_chars - 1].rsplit(" ", 1)[0] + "…"
    return blurb or title


def fetch_video_info(url: str, *, api_key: str | None = None) -> dict[str, object]:
    key = api_key or os.getenv("YOUTUBE_API_KEY", "")
    if not key:
        raise YouTubeAPIError("YOUTUBE_API_KEY not set (see .env.example)")
    video_id = extract_video_id(url)
    params = {
        "part": "snippet,statistics",
        "id": video_id,
        "key": key,
    }
    with httpx.Client(timeout=20.0) as client:
        resp = client.get(_VIDEOS_API, params=params)
        if resp.status_code == 403:
            raise YouTubeAPIError(f"Quota or permission error: {resp.text[:300]}")
        resp.raise_for_status()
        items = resp.json().get("items", [])
    if not items:
        raise YouTubeAPIError("Video not found or unavailable")
    item = items[0]
    snippet = item.get("snippet", {})
    stats = item.get("statistics", {})
    title = snippet.get("title") or ""
    description = snippet.get("description") or ""
    thumbs = snippet.get("thumbnails", {})
    thumb = thumbs.get("maxres") or thumbs.get("high") or thumbs.get("medium") or {}
    view_raw = stats.get("viewCount")
    comment_raw = stats.get("commentCount")
    from engine.llm.summarize import summarize_video_llm

    summary, summary_source = summarize_video_llm(
        title,
        description,
        snippet.get("channelTitle") or "",
    )
    return {
        "video_id": video_id,
        "channel_id": snippet.get("channelId"),
        "title": title,
        "channel": snippet.get("channelTitle") or "",
        "description": description,
        "summary": summary,
        "summary_source": summary_source,
        "thumbnail_url": thumb.get("url") or f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
        "published_at": snippet.get("publishedAt"),
        "view_count": int(view_raw) if view_raw is not None else None,
        "comment_count": int(comment_raw) if comment_raw is not None else None,
    }


def fetch_comments(
    url: str,
    *,
    max_comments: int = 500,
    api_key: str | None = None,
) -> list[CommentInput]:
    key = api_key or os.getenv("YOUTUBE_API_KEY", "")
    if not key:
        raise YouTubeAPIError("YOUTUBE_API_KEY not set (see .env.example)")

    video_id = extract_video_id(url)
    comments: list[CommentInput] = []
    page_token: str | None = None

    with httpx.Client(timeout=30.0) as client:
        while len(comments) < max_comments:
            params: dict[str, Any] = {
                "part": "snippet,replies",
                "videoId": video_id,
                "maxResults": min(100, max_comments - len(comments)),
                "key": key,
                "textFormat": "plainText",
                "order": "relevance",
            }
            if page_token:
                params["pageToken"] = page_token
            resp = client.get(_API, params=params)
            if resp.status_code == 403:
                raise YouTubeAPIError(f"Quota or permission error: {resp.text[:300]}")
            resp.raise_for_status()
            data = resp.json()
            for item in data.get("items", []):
                top = item["snippet"]["topLevelComment"]["snippet"]
                cid = item["snippet"]["topLevelComment"]["id"]
                comments.append(
                    CommentInput(
                        id=cid,
                        text=top.get("textDisplay") or top.get("textOriginal", ""),
                        author=top.get("authorDisplayName"),
                        author_channel_id=normalize_author_channel_id(top.get("authorChannelId")),
                    )
                )
                replies = item.get("replies", {}).get("comments", [])
                for rep in replies:
                    sn = rep["snippet"]
                    comments.append(
                        CommentInput(
                            id=rep["id"],
                            text=sn.get("textDisplay") or sn.get("textOriginal", ""),
                            author=sn.get("authorDisplayName"),
                            author_channel_id=normalize_author_channel_id(sn.get("authorChannelId")),
                            parent_id=cid,
                        )
                    )
                if len(comments) >= max_comments:
                    break
            page_token = data.get("nextPageToken")
            if not page_token:
                break
    return comments[:max_comments]


def extract_owner_reply_texts(
    comments: list[CommentInput],
    *,
    owner_channel_id: str | None,
    max_samples: int = 25,
) -> list[str]:
    if not owner_channel_id:
        return []
    texts: list[str] = []
    for c in comments:
        if c.author_channel_id == owner_channel_id:
            t = c.text.strip()
            if t:
                texts.append(t)
        if len(texts) >= max_samples:
            break
    return texts