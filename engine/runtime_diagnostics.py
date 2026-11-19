"""Startup/runtime diagnostics for the moderation stack."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from engine.llm.config import load_openrouter_config

logger = logging.getLogger(__name__)


def _detoxify_status() -> dict[str, Any]:
    try:
        import detoxify  # noqa: F401

        import_ok = True
        err = None
    except ImportError as exc:
        import_ok = False
        err = str(exc)
    torch_ok = False
    torch_err: str | None = None
    try:
        import torch

        torch_ok = True
        _ = torch.__version__
    except ImportError as exc:
        torch_err = str(exc)
    return {
        "detoxify_importable": import_ok,
        "detoxify_error": err,
        "torch_importable": torch_ok,
        "torch_error": torch_err,
    }


def _llm_config_status() -> dict[str, Any]:
    key_set = bool(os.getenv("OPENROUTER_API_KEY", "").strip())
    models: tuple[str, ...] = ()
    models_error: str | None = None
    if key_set:
        try:
            cfg = load_openrouter_config()
            models = cfg.models
        except ValueError as exc:
            models_error = str(exc)
    summary_model = os.getenv("OPENROUTER_SUMMARY_MODEL", "").strip() or None
    return {
        "openrouter_api_key_set": key_set,
        "openrouter_models": list(models),
        "openrouter_models_error": models_error,
        "openrouter_summary_model": summary_model,
        "use_llm": os.getenv("USE_LLM", "false").lower() == "true",
    }


def collect_runtime_status(*, probe_lr: bool = True) -> dict[str, Any]:
    from engine.pipeline import pipeline_from_env

    use_transformer = os.getenv("USE_TRANSFORMER", "true").lower() == "true"
    lr_path = os.getenv("LR_MODEL_PATH", "models/lr/v1/model.joblib")
    thresh_path = os.getenv("LR_THRESHOLDS_PATH", "models/lr/v1/thresholds.json")

    status: dict[str, Any] = {
        "use_transformer": use_transformer,
        "lr_model_path": str(Path(lr_path)),
        "lr_model_exists": Path(lr_path).is_file(),
        "lr_thresholds_path": str(Path(thresh_path)),
        "lr_thresholds_exists": Path(thresh_path).is_file(),
        "t_low": None,
        "t_high": None,
        "lr_probe_ok": False,
        **_llm_config_status(),
        **_detoxify_status(),
        "transformer_runtime_ok": False,
        "transformer_runtime_error": None,
        "llm_layer_configured": False,
        "llm_layer_error": None,
    }

    if probe_lr and status["lr_model_exists"]:
        try:
            pipe = pipeline_from_env()
            pipe.lr.harmful_probability("health check")
            status["lr_probe_ok"] = True
            status["t_low"] = pipe.lr.t_low
            status["t_high"] = pipe.lr.t_high
            if use_transformer and status["detoxify_importable"]:
                if os.getenv("TRANSFORMER_WARMUP", "false").lower() == "true":
                    try:
                        pipe.transformer.predict("health check")
                        status["transformer_runtime_ok"] = True
                    except Exception as exc:
                        status["transformer_runtime_error"] = str(exc)
                else:
                    status["transformer_runtime_ok"] = True
                    status["transformer_warmup"] = "deferred"
            elif use_transformer and not status["detoxify_importable"]:
                status["transformer_runtime_error"] = (
                    "USE_TRANSFORMER=true but detoxify not installed"
                )
            if status["use_llm"] and status["openrouter_api_key_set"]:
                try:
                    from engine.llm_layer import LLMLayer

                    LLMLayer(enabled=True)._client_or_raise()
                    status["llm_layer_configured"] = True
                except Exception as exc:
                    status["llm_layer_error"] = str(exc)
        except FileNotFoundError as exc:
            status["lr_probe_error"] = str(exc)
        except Exception as exc:
            status["lr_probe_error"] = str(exc)

    return status


def log_runtime_status() -> dict[str, Any]:
    status = collect_runtime_status()
    logger.info("Moderation runtime status: %s", status)
    for line in _format_lines(status):
        logger.info("%s", line)
    return status


def _format_lines(status: dict[str, Any]) -> list[str]:
    return [
        f"LR model: {status['lr_model_path']} (exists={status['lr_model_exists']})",
        f"Thresholds: t_low={status['t_low']} t_high={status['t_high']}",
        f"USE_TRANSFORMER={status['use_transformer']} "
        f"detoxify={status['detoxify_importable']} "
        f"transformer_ok={status['transformer_runtime_ok']}",
        f"USE_LLM={status['use_llm']} "
        f"openrouter_key={status['openrouter_api_key_set']} "
        f"models={status['openrouter_models']}",
    ]