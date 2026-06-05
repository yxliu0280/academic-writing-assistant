from __future__ import annotations

import os
import re
import tomllib
from dataclasses import replace
from pathlib import Path
from typing import Dict

from core.schemas import AppConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_CONFIG_FILENAME = "model_config.toml"
MODEL_CONFIG_TEMPLATE_FILENAME = "model_config.example.toml"
_LOCAL_RUNTIME_CONFIG_LOADED = False
_ENV_VAR_PATTERN = re.compile(r"\$(\w+)|\$\{([^}]+)\}")


def ensure_local_runtime_config_loaded(force: bool = False, base_dir: Path | None = None) -> None:
    """Load project-local runtime config once.

    Precedence:
    1. Existing process environment variables
    2. model_config.toml
    3. .env.local
    4. .env
    """

    global _LOCAL_RUNTIME_CONFIG_LOADED
    if _LOCAL_RUNTIME_CONFIG_LOADED and not force:
        return

    root_dir = Path(base_dir) if base_dir is not None else PROJECT_ROOT

    _ensure_model_config_file(root_dir)

    merged: Dict[str, str] = {}
    for filename in (".env", ".env.local"):
        path = root_dir / filename
        if not path.is_file():
            continue
        merged.update(_parse_env_file(path, context={**os.environ, **merged}))

    model_config_path = root_dir / MODEL_CONFIG_FILENAME
    if model_config_path.is_file():
        merged.update(
            _parse_model_config_file(model_config_path, context={**os.environ, **merged})
        )

    for key, value in merged.items():
        if not str(value).strip():
            continue
        os.environ.setdefault(key, value)

    _LOCAL_RUNTIME_CONFIG_LOADED = True


def _ensure_model_config_file(root_dir: Path) -> None:
    target = root_dir / MODEL_CONFIG_FILENAME
    if target.exists():
        return
    template = root_dir / MODEL_CONFIG_TEMPLATE_FILENAME
    if not template.is_file():
        return
    try:
        target.write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
    except OSError:
        return


def _parse_model_config_file(path: Path, context: Dict[str, str]) -> Dict[str, str]:
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    provider = str(payload.get("provider", "") or payload.get("backend", "")).strip()
    text_provider = str(payload.get("text_provider", "")).strip() or provider
    vlm_provider = str(payload.get("multimodal_provider", "")).strip() or provider

    api_key = str(payload.get("api_key", "")).strip()
    text_api_key = str(payload.get("text_api_key", "")).strip() or api_key
    vlm_api_key = str(payload.get("multimodal_api_key", "")).strip() or api_key

    text_model = _expand_env_value(str(payload.get("text_model", "")).strip(), context)
    vlm_model = _expand_env_value(
        str(payload.get("multimodal_model", "")).strip() or text_model,
        {**context, "text_model": text_model},
    )

    base_url = _expand_env_value(str(payload.get("base_url", "")).strip(), context)
    vlm_base_url = _expand_env_value(
        str(payload.get("multimodal_base_url", "")).strip() or base_url,
        {**context, "base_url": base_url},
    )

    mapped: Dict[str, str] = {}
    if text_provider:
        mapped["TEXT_LLM_PROVIDER"] = text_provider
    if vlm_provider:
        mapped["VLM_PROVIDER"] = vlm_provider
    if api_key:
        mapped["API_KEY"] = api_key
    if text_api_key:
        mapped["TEXT_API_KEY"] = text_api_key
    if vlm_api_key:
        mapped["VLM_API_KEY"] = vlm_api_key
    if text_model:
        mapped["MODEL_NAME"] = text_model
    if vlm_model:
        mapped["VLM_MODEL_NAME"] = vlm_model
    if base_url:
        mapped["BASE_URL"] = base_url
    if vlm_base_url:
        mapped["VLM_BASE_URL"] = vlm_base_url

    for field, env_key in {
        "timeout": "TIMEOUT",
        "max_retries": "MAX_RETRIES",
        "debug": "DEBUG",
        "llm_cache_enabled": "LLM_CACHE_ENABLED",
        "llm_cache_dir": "LLM_CACHE_DIR",
    }.items():
        value = payload.get(field)
        if value is None:
            continue
        mapped[env_key] = str(value)

    return mapped


def _parse_env_file(path: Path, context: Dict[str, str]) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue

        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue

        value = _normalize_env_value(raw_value.strip())
        resolved = _expand_env_value(value, {**context, **values})
        values[key] = resolved
    return values


def _normalize_env_value(value: str) -> str:
    if not value:
        return ""
    if value[0] in {"'", '"'} and value[-1:] == value[0]:
        inner = value[1:-1]
        if value[0] == '"':
            return bytes(inner, "utf-8").decode("unicode_escape")
        return inner

    comment_split = re.split(r"\s+#", value, maxsplit=1)
    return comment_split[0].strip()


def _expand_env_value(value: str, context: Dict[str, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        key = match.group(1) or match.group(2) or ""
        return str(context.get(key, os.getenv(key, "")))

    return _ENV_VAR_PATTERN.sub(repl, value)


def load_config_from_env(base: AppConfig | None = None) -> AppConfig:
    ensure_local_runtime_config_loaded()
    config = base or AppConfig()

    text_provider_name = os.getenv("TEXT_LLM_PROVIDER", config.text_provider_name)
    vlm_provider_name = os.getenv("VLM_PROVIDER", config.vlm_provider_name)

    table_abs_tolerance = float(
        os.getenv("TABLE_ABS_TOLERANCE", str(config.table_abs_tolerance))
    )
    table_rel_tolerance = float(
        os.getenv("TABLE_REL_TOLERANCE", str(config.table_rel_tolerance))
    )
    figure_abs_tolerance = float(
        os.getenv("FIGURE_ABS_TOLERANCE", str(config.figure_abs_tolerance))
    )
    figure_confidence_threshold = float(
        os.getenv(
            "FIGURE_CONFIDENCE_THRESHOLD", str(config.figure_confidence_threshold)
        )
    )
    style_protection_edit_ratio = float(
        os.getenv(
            "STYLE_PROTECTION_EDIT_RATIO", str(config.style_protection_edit_ratio)
        )
    )
    style_protection_sentence_changes = int(
        os.getenv(
            "STYLE_PROTECTION_SENTENCE_CHANGES",
            str(config.style_protection_sentence_changes),
        )
    )
    negotiation_max_rounds = int(
        os.getenv("NEGOTIATION_MAX_ROUNDS", str(config.negotiation_max_rounds))
    )

    providers_enabled = {
        "text_llm": text_provider_name.lower() != "none",
        "vlm": vlm_provider_name.lower() != "none",
    }

    return replace(
        config,
        text_provider_name=text_provider_name,
        vlm_provider_name=vlm_provider_name,
        table_abs_tolerance=table_abs_tolerance,
        table_rel_tolerance=table_rel_tolerance,
        figure_abs_tolerance=figure_abs_tolerance,
        figure_confidence_threshold=figure_confidence_threshold,
        style_protection_edit_ratio=style_protection_edit_ratio,
        style_protection_sentence_changes=style_protection_sentence_changes,
        negotiation_max_rounds=negotiation_max_rounds,
        providers_enabled=providers_enabled,
    )
