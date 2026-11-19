"""FastAPI backend for comment moderation."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api import db
from api.youtube_client import (
    YouTubeAPIError,
    extract_owner_reply_texts,
    extract_video_id,
    fetch_comments,
    fetch_video_info,
)
from engine.llm import config as _llm_config  # loads .env once
from engine.llm.config import load_openrouter_config, resolve_openrouter_bucket_models, resolve_openrouter_models
from engine.llm.prompt import BUCKET_ONE_LINERS
from engine.creator_buckets import apply_creator_buckets, filter_owner_comments
from engine.pipeline import ModerationPipeline, pipeline_from_env
from engine.runtime_diagnostics import collect_runtime_status, log_runtime_status

_llm_config  # used for side effect

logger = logging.getLogger(__name__)
_runtime_status: dict[str, object] = {}
from engine.llm.cache import get_daily_requests
from engine.llm.comment_insights import CommentForInsights, generate_comment_insights
from engine.llm.comment_qa import CommentForQA, ask_about_comments
from engine.llm.reply_assistant import suggest_replies
from engine.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ClusteredQuestionOut,
    CommentAskRequest,
    CommentAskResponse,
    CommentInsightsRequest,
    CommentInsightsResponse,
    SentimentInsightOut,
    JobResponse,
    JobStatus,
    ReplyItemIn,
    ReplySuggestionsRequest,
    ReplySuggestionsResponse,
    ReplySuggestionOut,
    YouTubeAnalyzeRequest,
    YouTubeVideoInfo,
)

_pipeline: ModerationPipeline | None = None


def get_pipeline() -> ModerationPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = pipeline_from_env()
    return _pipeline


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _runtime_status
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    db.init_db()
    _runtime_status = log_runtime_status()
    yield


API_FEATURE_VERSION = "0.2.0"

app = FastAPI(title="Comment Moderator API", version=API_FEATURE_VERSION, lifespan=lifespan)

_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000,http://127.0.0.1:3001",
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins if o.strip()],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, object]:
    live = collect_runtime_status(probe_lr=False)
    lr_ok = False
    try:
        get_pipeline().lr.harmful_probability("health check")
        lr_ok = True
        live["t_low"] = get_pipeline().lr.t_low
        live["t_high"] = get_pipeline().lr.t_high
    except FileNotFoundError:
        pass
    route_paths = sorted(
        {
            getattr(r, "path", "")
            for r in app.routes
            if getattr(r, "path", None) and "{" not in r.path and r.path.startswith("/")
        }
    )
    from engine.llm.backends import (
        llm_backend,
        llm_configured,
        probe_local_llm,
        resolve_llm_bucket_models,
        resolve_llm_moderation_models,
    )

    cfg = load_openrouter_config()
    mod_models = resolve_llm_moderation_models(cfg)
    bucket_models = resolve_llm_bucket_models(cfg)
    llm_ok = llm_configured()
    moderation_llm_enabled = (
        os.getenv("USE_LLM", "false").lower() == "true" and llm_ok and bool(mod_models)
    )
    bucket_llm_enabled = (
        os.getenv("USE_CREATOR_BUCKETS_LLM", "true").lower() == "true"
        and llm_ok
        and bool(bucket_models)
    )
    return {
        "status": "ok",
        "api_feature_version": API_FEATURE_VERSION,
        "routes": route_paths,
        "lr_model_loaded": lr_ok,
        "moderation_llm_enabled": moderation_llm_enabled,
        "bucket_llm_enabled": bucket_llm_enabled,
        "llm_backend": llm_backend(),
        "local_llm_reachable": probe_local_llm() if llm_backend() == "local" else None,
        "openrouter_moderation_models": list(mod_models),
        "openrouter_bucket_models": list(bucket_models),
        "bucket_definitions": BUCKET_ONE_LINERS,
        "runtime": {**_runtime_status, **live},
        "openrouter_budget": {"daily_used": get_daily_requests()},
        "ambiguous_escalation": {
            "detoxify_ready": bool(live.get("transformer_runtime_ok"))
            or (
                live.get("use_transformer")
                and live.get("detoxify_importable")
                and live.get("transformer_warmup") == "deferred"
            ),
            "llm_ready": bool(live.get("llm_layer_configured")),
            "note": (
                "Ambiguous LR band uses Detoxify when USE_TRANSFORMER=true; "
                "LLM when USE_LLM=true and models are set."
            ),
        },
    }


@app.get("/youtube/video", response_model=YouTubeVideoInfo)
def youtube_video(url: str) -> YouTubeVideoInfo:
    try:
        data = fetch_video_info(url)
    except YouTubeAPIError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return YouTubeVideoInfo.model_validate(data)


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    pipeline = get_pipeline()
    comments = req.comments
    if not req.include_replies:
        comments = [c for c in comments if not c.parent_id]
    result = pipeline.analyze_batch(comments)
    return apply_creator_buckets(result, comments)


def _run_youtube_job(job_id: str, url: str, max_comments: int) -> None:
    try:
        meta = fetch_video_info(url)
        all_comments = fetch_comments(url, max_comments=max_comments)
        owner_id = meta.get("channel_id")
        owner_samples = extract_owner_reply_texts(all_comments, owner_channel_id=owner_id)
        comments, excluded_owner = filter_owner_comments(
            all_comments, owner_channel_id=owner_id
        )
        result = get_pipeline().analyze_batch(comments)
        result.summary["excluded_owner_comments"] = excluded_owner
        result.summary["channel_id"] = owner_id
        result.summary["owner_reply_samples"] = owner_samples
        result.summary["owner_reply_count"] = len(owner_samples)
        result = apply_creator_buckets(result, comments)
        db.update_job(job_id, status=JobStatus.COMPLETED, result=result)
    except Exception as exc:
        db.update_job(job_id, status=JobStatus.FAILED, error=str(exc))


@app.post("/analyze/youtube", response_model=JobResponse)
def analyze_youtube(
    req: YouTubeAnalyzeRequest,
    background: BackgroundTasks,
) -> JobResponse:
    job_id = db.create_job()
    db.update_job(job_id, status=JobStatus.RUNNING)
    try:
        extract_video_id(req.url)
    except YouTubeAPIError as exc:
        db.update_job(job_id, status=JobStatus.FAILED, error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    background.add_task(_run_youtube_job, job_id, req.url, req.max_comments)
    return JobResponse(id=job_id, status=JobStatus.RUNNING)


@app.post("/youtube/reply-suggestions", response_model=ReplySuggestionsResponse)
def youtube_reply_suggestions(req: ReplySuggestionsRequest) -> ReplySuggestionsResponse:
    owner_samples = list(req.owner_reply_samples)
    channel = req.channel or ""
    try:
        if not owner_samples:
            try:
                meta = fetch_video_info(req.url)
                channel = channel or str(meta.get("channel") or "")
                all_comments = fetch_comments(req.url, max_comments=300)
                owner_id = meta.get("channel_id")
                owner_samples = extract_owner_reply_texts(
                    all_comments, owner_channel_id=str(owner_id) if owner_id else None
                )
            except YouTubeAPIError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        items = [ReplyItemIn(id=i.id, text=i.text, creator_bucket=i.creator_bucket) for i in req.items]
        result = suggest_replies(
            channel_name=channel or "the creator",
            owner_samples=owner_samples,
            items=items,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("reply-suggestions failed")
        raise HTTPException(status_code=500, detail=str(exc)[:300]) from exc
    return ReplySuggestionsResponse(
        suggestions=[
            ReplySuggestionOut(id=s.id, suggested_reply=s.suggested_reply, notes=s.notes)
            for s in result.suggestions
        ],
        llm_requests=result.llm_requests,
        cache_hits=result.cache_hits,
        error=result.error,
        budget_exhausted=result.budget_exhausted,
    )


@app.post("/youtube/comment-insights", response_model=CommentInsightsResponse)
def youtube_comment_insights(req: CommentInsightsRequest) -> CommentInsightsResponse:
    video_id = req.video_id or ""
    title = req.title or ""
    channel = req.channel or ""
    if req.url and not video_id:
        try:
            video_id = extract_video_id(req.url)
        except YouTubeAPIError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    if req.url and (not title or not channel):
        try:
            meta = fetch_video_info(req.url)
            title = title or str(meta.get("title") or "")
            channel = channel or str(meta.get("channel") or "")
            video_id = video_id or str(meta.get("video_id") or "")
        except YouTubeAPIError:
            pass
    if not video_id:
        video_id = "unknown"

    comments = [
        CommentForInsights(id=c.id, text=c.text, creator_bucket=c.creator_bucket) for c in req.comments
    ]
    result = generate_comment_insights(
        video_id=video_id,
        title=title,
        channel=channel,
        comments=comments,
        bucket_counts=req.bucket_counts,
    )
    return CommentInsightsResponse(
        themes=result.themes,
        questions=[
            ClusteredQuestionOut(
                theme=q.theme,
                examples=q.examples,
                count_estimate=q.count_estimate,
            )
            for q in result.questions
        ],
        requests=result.requests,
        sentiment=SentimentInsightOut(
            overall=result.sentiment.overall,
            summary=result.sentiment.summary,
        ),
        source=result.source,
        llm_requests=result.llm_requests,
        cache_hit=result.cache_hit,
        budget_exhausted=result.budget_exhausted,
        error=result.error,
    )


@app.post("/youtube/comment-ask", response_model=CommentAskResponse)
def youtube_comment_ask(req: CommentAskRequest) -> CommentAskResponse:
    video_id = req.video_id or ""
    title = req.title or ""
    channel = req.channel or ""
    if req.url and not video_id:
        try:
            video_id = extract_video_id(req.url)
        except YouTubeAPIError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    if req.url and (not title or not channel):
        try:
            meta = fetch_video_info(req.url)
            title = title or str(meta.get("title") or "")
            channel = channel or str(meta.get("channel") or "")
            video_id = video_id or str(meta.get("video_id") or "")
        except YouTubeAPIError:
            pass
    if not video_id:
        video_id = "unknown"

    comments = [CommentForQA(id=c.id, text=c.text, creator_bucket=c.creator_bucket) for c in req.comments]
    result = ask_about_comments(
        video_id=video_id,
        title=title,
        channel=channel,
        comments=comments,
        question=req.question,
    )
    return CommentAskResponse(
        answer=result.answer,
        citation_ids=result.citation_ids,
        source=result.source,
        llm_requests=result.llm_requests,
        cache_hit=result.cache_hit,
        budget_exhausted=result.budget_exhausted,
        error=result.error,
    )


@app.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str) -> JobResponse:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def run() -> None:
    import uvicorn

    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("api.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run()