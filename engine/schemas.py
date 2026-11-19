"""Pydantic models for moderation I/O."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class ModerationLabel(str, Enum):
    PRAISE = "praise"
    QUESTION = "question"
    NEUTRAL = "neutral"
    CRITIQUE = "critique"
    SPAM = "spam"
    TOXIC = "toxic"
    THREAT = "threat"
    IDENTITY_ATTACK = "identity_attack"


class ScoreBreakdown(BaseModel):
    toxicity: float = 0.0
    insult: float = 0.0
    threat: float = 0.0
    identity_attack: float = 0.0
    spam: float = 0.0


class CommentInput(BaseModel):
    id: str
    text: str
    author: str | None = None
    author_channel_id: str | None = None
    parent_id: str | None = None


class ModerationResult(BaseModel):
    id: str
    text: str
    label: ModerationLabel
    creator_bucket: str | None = None
    scores: ScoreBreakdown
    risk: int = Field(ge=0, le=100)
    decided_by: Literal["rules", "lr", "transformer", "heuristic", "spam", "llm"]
    reasons: list[str] = Field(default_factory=list)
    needs_human: bool = False
    trace: dict[str, Any] = Field(default_factory=dict)


class AnalyzeRequest(BaseModel):
    comments: list[CommentInput]
    include_replies: bool = True


class AnalyzeResponse(BaseModel):
    results: list[ModerationResult]
    summary: dict[str, Any]


class YouTubeAnalyzeRequest(BaseModel):
    url: str
    max_comments: int = Field(default=500, ge=1, le=2000)


class YouTubeVideoInfo(BaseModel):
    video_id: str
    channel_id: str | None = None
    title: str
    channel: str
    description: str
    summary: str
    summary_source: Literal["llm", "extractive"] = "extractive"
    thumbnail_url: str
    published_at: str | None = None
    view_count: int | None = None
    comment_count: int | None = None


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobResponse(BaseModel):
    id: str
    status: JobStatus
    error: str | None = None
    result: AnalyzeResponse | None = None


class ReplyItemIn(BaseModel):
    id: str
    text: str
    creator_bucket: str | None = None


class ReplySuggestionsRequest(BaseModel):
    url: str
    channel: str | None = None
    owner_reply_samples: list[str] = Field(default_factory=list, max_length=50)
    items: list[ReplyItemIn] = Field(min_length=1, max_length=30)


class ReplySuggestionOut(BaseModel):
    id: str
    suggested_reply: str
    notes: list[str] = Field(default_factory=list)


class ReplySuggestionsResponse(BaseModel):
    suggestions: list[ReplySuggestionOut]
    llm_requests: int = 0
    cache_hits: int = 0
    error: str | None = None
    budget_exhausted: bool = False


class CommentInsightInput(BaseModel):
    id: str
    text: str
    creator_bucket: str | None = None


class ClusteredQuestionOut(BaseModel):
    theme: str
    examples: list[str] = Field(default_factory=list)
    count_estimate: int | None = None


class SentimentInsightOut(BaseModel):
    overall: Literal["positive", "mixed", "negative", "neutral"] = "neutral"
    summary: str = ""


class CommentInsightsRequest(BaseModel):
    url: str = ""
    video_id: str | None = None
    title: str | None = None
    channel: str | None = None
    comments: list[CommentInsightInput] = Field(min_length=1, max_length=2000)
    bucket_counts: dict[str, int] | None = None


class CommentInsightsResponse(BaseModel):
    themes: list[str] = Field(default_factory=list)
    questions: list[ClusteredQuestionOut] = Field(default_factory=list)
    requests: list[str] = Field(default_factory=list)
    sentiment: SentimentInsightOut = Field(default_factory=SentimentInsightOut)
    source: Literal["llm", "heuristic"] = "heuristic"
    llm_requests: int = 0
    cache_hit: bool = False
    budget_exhausted: bool = False
    error: str | None = None


class CommentAskRequest(BaseModel):
    url: str = ""
    video_id: str | None = None
    title: str | None = None
    channel: str | None = None
    question: str = Field(min_length=1, max_length=500)
    comments: list[CommentInsightInput] = Field(min_length=1, max_length=2000)


class CommentAskResponse(BaseModel):
    answer: str = ""
    citation_ids: list[str] = Field(default_factory=list)
    source: Literal["llm", "heuristic"] = "heuristic"
    llm_requests: int = 0
    cache_hit: bool = False
    budget_exhausted: bool = False
    error: str | None = None