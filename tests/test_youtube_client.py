import pytest

from api.youtube_client import YouTubeAPIError, extract_video_id, normalize_author_channel_id


def test_extract_video_id() -> None:
    assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("https://m.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("https://www.youtube.com/watch?si=abc&v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_normalize_author_channel_id() -> None:
    assert normalize_author_channel_id("UCabc") == "UCabc"
    assert normalize_author_channel_id({"value": "UCabc"}) == "UCabc"
    assert normalize_author_channel_id(None) is None


def test_invalid_url() -> None:
    with pytest.raises(YouTubeAPIError):
        extract_video_id("https://example.com/not-youtube")