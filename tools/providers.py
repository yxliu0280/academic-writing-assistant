from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import time
from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional, Protocol, Type

import httpx

from core.config import ensure_local_runtime_config_loaded
from core.logging_utils import debug_enabled, get_logger, log_event, truncate_for_log
from tools.figure_utils import FigureFact

try:
    from pydantic import BaseModel, Field, ValidationError
except Exception:  # pragma: no cover
    BaseModel = object  # type: ignore[assignment]
    def Field(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        return kwargs.get("default", None)

    class ValidationError(Exception):
        pass


DEFAULT_ALIYUN_COMPAT_BASE_URL = (
    "https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1"
)
DEFAULT_ALIYUN_VLM_NATIVE_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
)


class LLMProviderError(Exception):
    def __init__(self, error_type: str, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.retryable = retryable


class TextLLMProvider(Protocol):
    def enabled(self) -> bool:
        ...

    def complete_json(
        self,
        schema: Type[BaseModel],
        messages: List[Dict[str, str]],
        request_id: str = "",
    ) -> Dict[str, Any]:
        ...

    def suggest_rewrite(self, sentence: str, context: Dict[str, object]) -> str:
        ...


class VLMProvider(Protocol):
    def enabled(self) -> bool:
        ...

    def extract_numeric_facts(
        self,
        image_bytes: bytes,
        meta: Dict[str, object],
    ) -> List[FigureFact]:
        ...

    def complete_json_from_image(
        self,
        schema: Type[BaseModel],
        prompt: str,
        image_bytes: bytes,
        meta: Dict[str, object],
        request_id: str = "",
    ) -> Dict[str, Any]:
        ...


@dataclass
class NullTextLLMProvider:
    def enabled(self) -> bool:
        return False

    def complete_json(
        self,
        schema: Type[BaseModel],
        messages: List[Dict[str, str]],
        request_id: str = "",
    ) -> Dict[str, Any]:
        raise LLMProviderError("disabled", "Text LLM provider is disabled.", retryable=False)

    def suggest_rewrite(self, sentence: str, context: Dict[str, object]) -> str:
        return ""


@dataclass
class MockTextLLMProvider:
    def enabled(self) -> bool:
        return True

    def complete_json(
        self,
        schema: Type[BaseModel],
        messages: List[Dict[str, str]],
        request_id: str = "",
    ) -> Dict[str, Any]:
        data = {"claims": []}
        if hasattr(schema, "model_validate"):
            validated = schema.model_validate(data)
            return validated.model_dump()
        return data

    def suggest_rewrite(self, sentence: str, context: Dict[str, object]) -> str:
        return f"Consider tightening this sentence: {sentence[:120]}"


@dataclass
class NullVLMProvider:
    def enabled(self) -> bool:
        return False

    def extract_numeric_facts(
        self,
        image_bytes: bytes,
        meta: Dict[str, object],
    ) -> List[FigureFact]:
        return []

    def complete_json_from_image(
        self,
        schema: Type[BaseModel],
        prompt: str,
        image_bytes: bytes,
        meta: Dict[str, object],
        request_id: str = "",
    ) -> Dict[str, Any]:
        raise LLMProviderError("disabled", "VLM provider is disabled.", retryable=False)


@dataclass
class MockVLMProvider:
    """Deterministic provider for demos/tests.

    Expected metadata field:
    - numeric_facts_json: JSON string containing list of facts:
      [{"figure_id":"2","value":88,"evidence_type":"label_read","confidence":0.93}]
    """

    def enabled(self) -> bool:
        return True

    def extract_numeric_facts(
        self,
        image_bytes: bytes,
        meta: Dict[str, object],
    ) -> List[FigureFact]:
        raw = str(meta.get("numeric_facts_json", "") or "")
        if not raw:
            return []
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return []

        facts: List[FigureFact] = []
        for item in payload:
            facts.append(
                FigureFact(
                    figure_id=str(item.get("figure_id", "")),
                    value=float(item.get("value", 0.0)),
                    evidence_type=str(item.get("evidence_type", "unknown")),
                    confidence=float(item.get("confidence", 0.0)),
                    source="mock_vlm",
                    metric=str(item.get("metric", "")),
                    subject=str(item.get("subject", "ours")),
                    unit=str(item.get("unit", "raw")),
                    scale=str(item.get("scale", "unknown")),
                    series_name=str(item.get("series_name", "")),
                    x_value=float(item["x_value"]) if item.get("x_value") is not None else None,
                    x_unit=str(item.get("x_unit", "")),
                    extra=dict(item.get("extra", {})) if isinstance(item.get("extra"), dict) else {},
                )
            )
        return facts

    def complete_json_from_image(
        self,
        schema: Type[BaseModel],
        prompt: str,
        image_bytes: bytes,
        meta: Dict[str, object],
        request_id: str = "",
    ) -> Dict[str, Any]:
        raw = str(meta.get("raw_judge_json", "") or "")
        if not raw:
            raise LLMProviderError("disabled", "Mock raw VLM response not configured.", retryable=False)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMProviderError("json_parse_error", str(exc), retryable=False) from exc
        if hasattr(schema, "model_validate"):
            validated = schema.model_validate(payload)
            return validated.model_dump()
        return payload


class AliyunVLMFactsItem(BaseModel):
    figure_id: str
    value: float
    evidence_type: str = "unknown"
    confidence: float
    metric: Optional[str] = ""
    subject: Optional[str] = "ours"
    unit: Optional[str] = "raw"
    scale: Optional[str] = "unknown"
    series_name: Optional[str] = ""
    x_value: Optional[float] = None
    x_unit: Optional[str] = ""
    extra: Dict[str, Any] = Field(default_factory=dict)


class AliyunVLMFactsResponse(BaseModel):
    facts: List[AliyunVLMFactsItem] = Field(default_factory=list)


class _AliyunResponsesClient:
    def __init__(
        self,
        *,
        base_url_env: str,
        model_env: str,
        api_key_env: str,
        provider_name: str,
        timeout_env: str = "TIMEOUT",
        max_retries_env: str = "MAX_RETRIES",
    ) -> None:
        ensure_local_runtime_config_loaded()
        self.logger = get_logger(__name__)
        self.provider_name = provider_name
        self.base_url = (
            os.getenv(base_url_env)
            or os.getenv("BASE_URL")
            or DEFAULT_ALIYUN_COMPAT_BASE_URL
        ).strip()
        self.api_key = (
            os.getenv(api_key_env)
            or os.getenv("API_KEY")
            or os.getenv("DASHSCOPE_API_KEY")
            or ""
        ).strip()
        self.model_name = (os.getenv(model_env) or os.getenv("MODEL_NAME") or "").strip()
        self.timeout = float(os.getenv(timeout_env, "30"))
        self.max_retries = max(1, int(os.getenv(max_retries_env, "3")))

        self.client = None
        self._openai_errors: Dict[str, Any] = {}
        self._dependency_available = True
        self._init_client()

    def _init_client(self) -> None:
        if not (self.api_key and self.model_name):
            return
        try:
            from openai import (
                APIConnectionError,
                APIStatusError,
                APITimeoutError,
                AuthenticationError,
                OpenAI,
                RateLimitError,
            )
        except Exception:
            self._dependency_available = False
            return

        kwargs: Dict[str, Any] = {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "timeout": self.timeout,
        }
        self.client = OpenAI(**kwargs)
        self._openai_errors = {
            "APIConnectionError": APIConnectionError,
            "APIStatusError": APIStatusError,
            "APITimeoutError": APITimeoutError,
            "AuthenticationError": AuthenticationError,
            "RateLimitError": RateLimitError,
        }

    @staticmethod
    def _parse_env_bool(raw: str | None) -> Optional[bool]:
        text = str(raw or "").strip().lower()
        if not text:
            return None
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        return None

    def _thinking_override(self) -> Optional[bool]:
        # Unified runtime knob to force on/off thinking behavior for compatible providers.
        # Priority: ENABLE_THINKING > ALIYUN_ENABLE_THINKING.
        primary = self._parse_env_bool(os.getenv("ENABLE_THINKING"))
        if primary is not None:
            return primary
        return self._parse_env_bool(os.getenv("ALIYUN_ENABLE_THINKING"))

    def enabled(self) -> bool:
        return self.client is not None and bool(self.api_key) and bool(self.model_name)

    def _map_openai_error(self, exc: Exception) -> LLMProviderError:
        timeout_cls = self._openai_errors.get("APITimeoutError")
        conn_cls = self._openai_errors.get("APIConnectionError")
        auth_cls = self._openai_errors.get("AuthenticationError")
        rate_cls = self._openai_errors.get("RateLimitError")
        status_cls = self._openai_errors.get("APIStatusError")

        if timeout_cls and isinstance(exc, timeout_cls):
            return LLMProviderError("timeout_error", str(exc), retryable=True)
        if conn_cls and isinstance(exc, conn_cls):
            return LLMProviderError("network_error", str(exc), retryable=True)
        if rate_cls and isinstance(exc, rate_cls):
            return LLMProviderError("api_error", str(exc), retryable=True)
        if auth_cls and isinstance(exc, auth_cls):
            return LLMProviderError("auth_error", str(exc), retryable=False)
        if status_cls and isinstance(exc, status_cls):
            return LLMProviderError("api_error", str(exc), retryable=True)
        return LLMProviderError("unknown_error", str(exc), retryable=False)

    @staticmethod
    def _backoff(attempt: int) -> None:
        time.sleep(min(0.5 * (2 ** (attempt - 1)), 8.0))

    def _parse_json_text(self, text: str) -> Dict[str, Any]:
        raw = text.strip()
        if not raw:
            raise LLMProviderError("json_parse_error", "Model returned empty text.", retryable=True)

        candidates = [raw]

        fenced = re.findall(r"```(?:json)?\s*(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
        candidates.extend([c.strip() for c in fenced if c.strip()])

        brace_match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if brace_match:
            candidates.append(brace_match.group(0).strip())

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue

        raise LLMProviderError(
            "json_parse_error",
            "Model output is not valid JSON object.",
            retryable=True,
        )

    @staticmethod
    def _messages_to_responses_input(messages: List[Dict[str, str]]) -> tuple[str, List[Dict[str, Any]]]:
        instructions_parts: List[str] = []
        payload: List[Dict[str, Any]] = []

        for message in messages:
            role = str(message.get("role", "user")).lower().strip()
            content = str(message.get("content", ""))
            if role == "system":
                instructions_parts.append(content)
                continue
            if role not in {"user", "assistant"}:
                role = "user"
            payload.append(
                {
                    "role": role,
                    "content": [{"type": "input_text", "text": content}],
                }
            )

        instructions = "\n\n".join([p for p in instructions_parts if p.strip()])
        return instructions, payload

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        output = getattr(response, "output", None)
        if output is None and isinstance(response, dict):
            output = response.get("output")

        chunks: List[str] = []
        for item in output or []:
            content = getattr(item, "content", None)
            if content is None and isinstance(item, dict):
                content = item.get("content", [])
            for part in content or []:
                text = getattr(part, "text", None)
                if text is None and isinstance(part, dict):
                    text = part.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunks).strip()

    @staticmethod
    def _input_payload_to_chat_messages(
        input_payload: List[Dict[str, Any]],
        instructions: str = "",
    ) -> List[Dict[str, Any]]:
        chat_messages: List[Dict[str, Any]] = []
        if instructions.strip():
            chat_messages.append({"role": "system", "content": instructions.strip()})

        for item in input_payload:
            role = str(item.get("role", "user") or "user").strip().lower()
            if role not in {"user", "assistant", "system"}:
                role = "user"
            content_parts = item.get("content", [])

            has_image = any(isinstance(part, dict) and part.get("type") == "input_image" for part in content_parts)
            if not has_image:
                text_parts: List[str] = []
                for part in content_parts:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "input_text":
                        text_parts.append(str(part.get("text", "")))
                chat_messages.append({"role": role, "content": "\n".join([p for p in text_parts if p.strip()])})
                continue

            multimodal_parts: List[Dict[str, Any]] = []
            for part in content_parts:
                if not isinstance(part, dict):
                    continue
                part_type = str(part.get("type", "")).strip().lower()
                if part_type == "input_text":
                    multimodal_parts.append({"type": "text", "text": str(part.get("text", ""))})
                    continue
                if part_type == "input_image":
                    image_url = str(part.get("image_url", ""))
                    if image_url:
                        multimodal_parts.append({"type": "image_url", "image_url": {"url": image_url}})
            chat_messages.append({"role": role, "content": multimodal_parts})
        return chat_messages

    def _responses_create(self, *, model: str, input_payload: List[Dict[str, Any]], instructions: str = "") -> str:
        if not self.client:
            raise LLMProviderError("config_error", "Client not initialized.", retryable=False)
        responses_api = getattr(self.client, "responses", None)
        thinking_override = self._thinking_override()

        if responses_api is not None and hasattr(responses_api, "create"):
            kwargs: Dict[str, Any] = {
                "model": model,
                "input": input_payload,
                "temperature": 0,
            }
            if instructions:
                kwargs["instructions"] = instructions
            if thinking_override is not None:
                kwargs["extra_body"] = {"enable_thinking": thinking_override}
            try:
                response = responses_api.create(**kwargs)
                return self._extract_response_text(response)
            except Exception as exc:
                raise self._map_openai_error(exc) from exc

        # SDK compatibility fallback: call /responses directly when the installed
        # openai client does not expose `client.responses`.
        manual_url = f"{self.base_url.rstrip('/')}/responses"
        manual_payload: Dict[str, Any] = {
            "model": model,
            "input": input_payload,
            "temperature": 0,
        }
        if instructions:
            manual_payload["instructions"] = instructions
        if thinking_override is not None:
            manual_payload["enable_thinking"] = thinking_override
        manual_error: Optional[LLMProviderError] = None
        try:
            response = httpx.post(
                manual_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=manual_payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            text = self._extract_response_text(data)
            if text:
                return text
            manual_error = LLMProviderError(
                "json_parse_error",
                "Manual /responses returned empty text.",
                retryable=True,
            )
        except httpx.TimeoutException as exc:
            manual_error = LLMProviderError("timeout_error", str(exc), retryable=True)
        except httpx.ConnectError as exc:
            manual_error = LLMProviderError("network_error", str(exc), retryable=True)
        except httpx.HTTPStatusError as exc:
            status_code = int(exc.response.status_code) if exc.response is not None else 0
            retryable = status_code >= 500 or status_code in {408, 429}
            manual_error = LLMProviderError("api_error", str(exc), retryable=retryable)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            manual_error = LLMProviderError("json_parse_error", str(exc), retryable=True)
        except Exception as exc:
            # Keep compatibility path resilient without masking unexpected issues.
            manual_error = LLMProviderError("unknown_error", str(exc), retryable=True)

        chat_completions = getattr(getattr(self.client, "chat", None), "completions", None)
        if chat_completions is not None and hasattr(chat_completions, "create"):
            chat_messages = self._input_payload_to_chat_messages(
                input_payload=input_payload,
                instructions=instructions,
            )
            try:
                kwargs: Dict[str, Any] = {
                    "model": model,
                    "messages": chat_messages,
                    "temperature": 0,
                }
                if thinking_override is not None:
                    kwargs["extra_body"] = {"enable_thinking": thinking_override}
                response = chat_completions.create(**kwargs)
                choices = getattr(response, "choices", None)
                if not choices and isinstance(response, dict):
                    choices = response.get("choices", [])
                if not choices:
                    raise LLMProviderError(
                        "json_parse_error",
                        "Model returned empty choices.",
                        retryable=True,
                    )
                first = choices[0]
                message = getattr(first, "message", None)
                if message is None and isinstance(first, dict):
                    message = first.get("message", {})
                content = getattr(message, "content", None)
                if content is None and isinstance(message, dict):
                    content = message.get("content", "")
                if isinstance(content, list):
                    text_parts: List[str] = []
                    for part in content:
                        if isinstance(part, dict):
                            text = part.get("text")
                            if isinstance(text, str):
                                text_parts.append(text)
                    content = "\n".join(text_parts)
                text_content = str(content or "").strip()
                if not text_content:
                    raise LLMProviderError(
                        "json_parse_error",
                        "Model returned empty message content.",
                        retryable=True,
                    )
                return text_content
            except Exception as exc:
                if isinstance(exc, LLMProviderError):
                    raise
                raise self._map_openai_error(exc) from exc

        if manual_error is not None:
            raise manual_error

        raise LLMProviderError(
            "dependency_missing",
            "OpenAI client does not support responses or chat.completions API.",
            retryable=False,
        )


class AliyunTextLLMProvider(_AliyunResponsesClient):
    def __init__(self) -> None:
        super().__init__(
            base_url_env="BASE_URL",
            model_env="MODEL_NAME",
            api_key_env="TEXT_API_KEY",
            provider_name="aliyun_text",
        )
        self.cache_enabled = os.getenv("LLM_CACHE_ENABLED", "0") == "1"
        self.cache_dir = Path(os.getenv("LLM_CACHE_DIR", ".cache/llm"))

    def complete_json(
        self,
        schema: Type[BaseModel],
        messages: List[Dict[str, str]],
        request_id: str = "",
    ) -> Dict[str, Any]:
        if not self.enabled():
            if not self._dependency_available:
                raise LLMProviderError(
                    "dependency_missing",
                    "openai package is not available. Install dependencies from requirements.txt.",
                    retryable=False,
                )
            raise LLMProviderError(
                "config_error",
                "Text provider is not configured. Set api_key and text_model in model_config.toml, or use environment variables.",
                retryable=False,
            )

        if not hasattr(schema, "model_validate"):
            raise LLMProviderError(
                "schema_type_error",
                "Schema must be a Pydantic model class with model_validate().",
                retryable=False,
            )

        if self.cache_enabled:
            cached = self._load_cache(schema=schema, messages=messages)
            if cached is not None:
                log_event(
                    self.logger,
                    logging.INFO,
                    "llm_cache_hit",
                    request_id=request_id,
                    model=self.model_name,
                    provider=self.provider_name,
                )
                return cached

        instructions, input_payload = self._messages_to_responses_input(messages)
        last_error: LLMProviderError | None = None

        for attempt in range(1, self.max_retries + 1):
            log_event(
                self.logger,
                logging.INFO,
                "llm_request_start",
                request_id=request_id,
                provider=self.provider_name,
                model=self.model_name,
                attempt=attempt,
                max_retries=self.max_retries,
            )

            if debug_enabled():
                log_event(
                    self.logger,
                    logging.DEBUG,
                    "llm_request_messages",
                    request_id=request_id,
                    messages=[
                        {
                            "role": m.get("role", ""),
                            "content": truncate_for_log(str(m.get("content", "")), max_len=1200),
                        }
                        for m in messages
                    ],
                )

            try:
                response_text = self._responses_create(
                    model=self.model_name,
                    input_payload=input_payload,
                    instructions=instructions,
                )

                if debug_enabled():
                    log_event(
                        self.logger,
                        logging.DEBUG,
                        "llm_response_raw",
                        request_id=request_id,
                        raw_text=truncate_for_log(response_text, max_len=1800),
                    )

                parsed = self._parse_json_text(response_text)
                validated = schema.model_validate(parsed)
                result = validated.model_dump()

                if debug_enabled():
                    log_event(
                        self.logger,
                        logging.DEBUG,
                        "llm_response_parsed",
                        request_id=request_id,
                        parsed_json=result,
                    )

                if self.cache_enabled:
                    self._save_cache(schema=schema, messages=messages, response_text=response_text)

                log_event(
                    self.logger,
                    logging.INFO,
                    "llm_request_success",
                    request_id=request_id,
                    provider=self.provider_name,
                    model=self.model_name,
                    attempt=attempt,
                )
                return result
            except LLMProviderError as exc:
                last_error = exc
                log_event(
                    self.logger,
                    logging.WARNING,
                    "llm_request_failure",
                    request_id=request_id,
                    provider=self.provider_name,
                    error_type=exc.error_type,
                    retryable=exc.retryable,
                    attempt=attempt,
                    message=str(exc),
                )
                if (not exc.retryable) or attempt >= self.max_retries:
                    break
                self._backoff(attempt)
            except ValidationError as exc:
                last_error = LLMProviderError("schema_validation_error", str(exc), retryable=True)
                log_event(
                    self.logger,
                    logging.WARNING,
                    "llm_request_failure",
                    request_id=request_id,
                    provider=self.provider_name,
                    error_type=last_error.error_type,
                    retryable=last_error.retryable,
                    attempt=attempt,
                    message=str(last_error),
                )
                if attempt >= self.max_retries:
                    break
                self._backoff(attempt)

        raise last_error or LLMProviderError(
            "unknown_error",
            "LLM request failed without explicit error.",
            retryable=False,
        )

    def suggest_rewrite(self, sentence: str, context: Dict[str, object]) -> str:
        return ""

    def _cache_key(self, schema: Type[BaseModel], messages: List[Dict[str, str]]) -> str:
        schema_payload = {}
        if hasattr(schema, "model_json_schema"):
            schema_payload = schema.model_json_schema()
        payload = {
            "provider": self.provider_name,
            "model": self.model_name,
            "schema": schema_payload,
            "messages": messages,
        }
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def _load_cache(
        self,
        schema: Type[BaseModel],
        messages: List[Dict[str, str]],
    ) -> Dict[str, Any] | None:
        key = self._cache_key(schema=schema, messages=messages)
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if "response_text" in payload:
                parsed = self._parse_json_text(str(payload["response_text"]))
                validated = schema.model_validate(parsed)
                return validated.model_dump()
        except (json.JSONDecodeError, ValidationError, LLMProviderError):
            return None
        return None

    def _save_cache(
        self,
        schema: Type[BaseModel],
        messages: List[Dict[str, str]],
        response_text: str,
    ) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        key = self._cache_key(schema=schema, messages=messages)
        path = self.cache_dir / f"{key}.json"
        payload = {
            "provider": self.provider_name,
            "model": self.model_name,
            "created_at": time.time(),
            "response_text": response_text,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class AliyunVLMProvider(_AliyunResponsesClient):
    def __init__(self) -> None:
        super().__init__(
            base_url_env="VLM_BASE_URL",
            model_env="VLM_MODEL_NAME",
            api_key_env="VLM_API_KEY",
            provider_name="aliyun_vlm",
        )
        self.native_vlm_url = (
            os.getenv("VLM_NATIVE_URL")
            or os.getenv("MULTIMODAL_GENERATION_URL")
            or DEFAULT_ALIYUN_VLM_NATIVE_URL
        ).strip()

    @staticmethod
    def _extract_dashscope_message_text(payload: Dict[str, Any]) -> str:
        output = payload.get("output", {})
        if not isinstance(output, dict):
            return ""

        choices = output.get("choices", [])
        if not isinstance(choices, list) or not choices:
            return ""

        first = choices[0]
        if not isinstance(first, dict):
            return ""

        message = first.get("message", {})
        if not isinstance(message, dict):
            return ""

        content = message.get("content", [])
        if isinstance(content, str):
            return content.strip()
        if not isinstance(content, list):
            return ""

        text_parts: List[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                text_parts.append(text.strip())
        return "\n".join(text_parts).strip()

    def _native_vlm_create(
        self,
        *,
        image_data: str,
        mime: str,
        prompt: str,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"text": prompt},
                            {"image": f"data:{mime};base64,{image_data}"},
                        ],
                    }
                ]
            },
            "parameters": {"result_format": "message"},
        }
        try:
            response = httpx.post(
                self.native_vlm_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException as exc:
            raise LLMProviderError("timeout_error", str(exc), retryable=True) from exc
        except httpx.ConnectError as exc:
            raise LLMProviderError("network_error", str(exc), retryable=True) from exc
        except httpx.HTTPStatusError as exc:
            status_code = int(exc.response.status_code) if exc.response is not None else 0
            retryable = status_code >= 500 or status_code in {408, 429}
            raise LLMProviderError("api_error", str(exc), retryable=retryable) from exc
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            raise LLMProviderError("json_parse_error", str(exc), retryable=True) from exc
        except Exception as exc:
            raise LLMProviderError("unknown_error", str(exc), retryable=True) from exc

        if isinstance(data.get("code"), str) and data.get("code"):
            message = str(data.get("message", data["code"]))
            retryable = str(data.get("code", "")).lower() in {"throttling", "internalerror"}
            raise LLMProviderError("api_error", message, retryable=retryable)

        text = self._extract_dashscope_message_text(data)
        if text:
            return text
        raise LLMProviderError(
            "json_parse_error",
            "DashScope multimodal endpoint returned empty text.",
            retryable=True,
        )

    def extract_numeric_facts(
        self,
        image_bytes: bytes,
        meta: Dict[str, object],
    ) -> List[FigureFact]:
        request_id = str(meta.get("request_id", ""))

        if not self.enabled() or not image_bytes:
            return []

        figure_id = self._guess_figure_id(meta)
        image_ext = self._resolve_image_ext(meta)
        mime = "image/jpeg" if image_ext in {"jpg", "jpeg"} else "image/png"
        image_data = base64.b64encode(image_bytes).decode("utf-8")

        prompt = (
            "Extract verifiable numeric facts from this chart image. "
            "Prioritize explicit labels/callouts. "
            "Return JSON only with schema:\n"
            "{\n"
            "  \"facts\": [\n"
            "    {\n"
            "      \"figure_id\": string,\n"
            "      \"value\": number,\n"
            "      \"evidence_type\": \"label_read\"|\"callout_read\"|\"legend_only\"|\"axis_estimate\"|\"unknown\",\n"
            "      \"confidence\": number,\n"
            "      \"metric\": string,\n"
            "      \"subject\": string,\n"
            "      \"unit\": \"raw\"|\"percent\"|\"pp\",\n"
            "      \"scale\": \"0_1\"|\"0_100\"|\"raw\"|\"unknown\",\n"
            "      \"series_name\": string,\n"
            "      \"x_value\": number|null,\n"
            "      \"x_unit\": string,\n"
            "      \"extra\": object\n"
            "    }\n"
            "  ]\n"
            "}\n"
            f"Use figure_id='{figure_id}'."
        )

        last_error: LLMProviderError | None = None
        for attempt in range(1, self.max_retries + 1):
            log_event(
                self.logger,
                logging.INFO,
                "vlm_request_start",
                request_id=request_id,
                provider=self.provider_name,
                model=self.model_name,
                attempt=attempt,
                max_retries=self.max_retries,
            )
            try:
                response_text = self._native_vlm_create(
                    image_data=image_data,
                    mime=mime,
                    prompt=prompt,
                )
                if debug_enabled():
                    log_event(
                        self.logger,
                        logging.DEBUG,
                        "vlm_response_raw",
                        request_id=request_id,
                        raw_text=truncate_for_log(response_text, max_len=1800),
                        transport="dashscope_multimodal_generation",
                    )

                parsed = self._parse_json_text(response_text)
                validated = AliyunVLMFactsResponse.model_validate(parsed)
                facts: List[FigureFact] = []
                for item in validated.facts:
                    evidence_type = str(item.evidence_type or "unknown").strip().lower()
                    if evidence_type not in {
                        "label_read",
                        "callout_read",
                        "legend_only",
                        "axis_estimate",
                        "unknown",
                    }:
                        evidence_type = "unknown"
                    facts.append(
                        FigureFact(
                            figure_id=item.figure_id,
                            value=float(item.value),
                            evidence_type=evidence_type,  # type: ignore[arg-type]
                            confidence=max(0.0, min(1.0, float(item.confidence))),
                            source="aliyun_vlm",
                            metric=str(item.metric or ""),
                            subject=str(item.subject or "ours"),
                            unit=str(item.unit or "raw"),
                            scale=str(item.scale or "unknown"),
                            series_name=str(item.series_name or ""),
                            x_value=item.x_value,
                            x_unit=str(item.x_unit or ""),
                            extra=dict(item.extra or {}),
                        )
                    )
                log_event(
                    self.logger,
                    logging.INFO,
                    "vlm_request_success",
                    request_id=request_id,
                    provider=self.provider_name,
                    model=self.model_name,
                    fact_count=len(facts),
                    transport="dashscope_multimodal_generation",
                )
                return facts
            except LLMProviderError as exc:
                last_error = exc
                log_event(
                    self.logger,
                    logging.WARNING,
                    "vlm_request_failure",
                    request_id=request_id,
                    provider=self.provider_name,
                    error_type=exc.error_type,
                    retryable=exc.retryable,
                    attempt=attempt,
                    message=str(exc),
                )
                if (not exc.retryable) or attempt >= self.max_retries:
                    break
                self._backoff(attempt)
            except ValidationError as exc:
                last_error = LLMProviderError("schema_validation_error", str(exc), retryable=True)
                log_event(
                    self.logger,
                    logging.WARNING,
                    "vlm_request_failure",
                    request_id=request_id,
                    provider=self.provider_name,
                    error_type=last_error.error_type,
                    retryable=True,
                    attempt=attempt,
                    message=str(last_error),
                )
                if attempt >= self.max_retries:
                    break
                self._backoff(attempt)

        if last_error:
            log_event(
                self.logger,
                logging.WARNING,
                "vlm_fallback_empty",
                request_id=request_id,
                provider=self.provider_name,
                reason=last_error.error_type,
            )
        return []

    def complete_json_from_image(
        self,
        schema: Type[BaseModel],
        prompt: str,
        image_bytes: bytes,
        meta: Dict[str, object],
        request_id: str = "",
    ) -> Dict[str, Any]:
        if not self.enabled():
            raise LLMProviderError(
                "config_error",
                "VLM provider is not configured. Set multimodal provider/model/api_key.",
                retryable=False,
            )
        if not image_bytes:
            raise LLMProviderError("input_error", "Image bytes are empty.", retryable=False)
        if not hasattr(schema, "model_validate"):
            raise LLMProviderError(
                "schema_type_error",
                "Schema must be a Pydantic model class with model_validate().",
                retryable=False,
            )

        image_ext = self._resolve_image_ext(meta)
        mime = "image/jpeg" if image_ext in {"jpg", "jpeg"} else "image/png"
        image_data = base64.b64encode(image_bytes).decode("utf-8")
        last_error: LLMProviderError | None = None

        for attempt in range(1, self.max_retries + 1):
            log_event(
                self.logger,
                logging.INFO,
                "vlm_json_request_start",
                request_id=request_id,
                provider=self.provider_name,
                model=self.model_name,
                attempt=attempt,
                max_retries=self.max_retries,
            )
            try:
                response_text = self._native_vlm_create(
                    image_data=image_data,
                    mime=mime,
                    prompt=prompt,
                )
                if debug_enabled():
                    log_event(
                        self.logger,
                        logging.DEBUG,
                        "vlm_json_response_raw",
                        request_id=request_id,
                        raw_text=truncate_for_log(response_text, max_len=1800),
                        transport="dashscope_multimodal_generation",
                    )
                parsed = self._parse_json_text(response_text)
                validated = schema.model_validate(parsed)
                result = validated.model_dump()
                log_event(
                    self.logger,
                    logging.INFO,
                    "vlm_json_request_success",
                    request_id=request_id,
                    provider=self.provider_name,
                    model=self.model_name,
                    attempt=attempt,
                )
                return result
            except LLMProviderError as exc:
                last_error = exc
                log_event(
                    self.logger,
                    logging.WARNING,
                    "vlm_json_request_failure",
                    request_id=request_id,
                    provider=self.provider_name,
                    error_type=exc.error_type,
                    retryable=exc.retryable,
                    attempt=attempt,
                    message=str(exc),
                )
                if (not exc.retryable) or attempt >= self.max_retries:
                    break
                self._backoff(attempt)
            except ValidationError as exc:
                last_error = LLMProviderError("schema_validation_error", str(exc), retryable=True)
                log_event(
                    self.logger,
                    logging.WARNING,
                    "vlm_json_request_failure",
                    request_id=request_id,
                    provider=self.provider_name,
                    error_type=last_error.error_type,
                    retryable=True,
                    attempt=attempt,
                    message=str(last_error),
                )
                if attempt >= self.max_retries:
                    break
                self._backoff(attempt)

        raise last_error or LLMProviderError(
            "unknown_error",
            "VLM JSON request failed without explicit error.",
            retryable=False,
        )

    @staticmethod
    def _guess_figure_id(meta: Dict[str, object]) -> str:
        if meta.get("figure_id"):
            return str(meta["figure_id"])
        name = str(meta.get("name", ""))
        m = re.search(r"(\d+)", name)
        if m:
            return m.group(1)
        return "unknown"

    @staticmethod
    def _resolve_image_ext(meta: Dict[str, object]) -> str:
        ext = str(meta.get("ext", "")).strip().lower().lstrip(".")
        if not ext:
            name = str(meta.get("name", "")).strip()
            ext = Path(name).suffix.lower().lstrip(".")
        ext = ext.replace("jpeg", "jpg")
        return ext or "png"


@lru_cache(maxsize=8)
def build_text_provider(name: str) -> TextLLMProvider:
    lname = (name or "none").lower().strip()
    if lname == "mock":
        return MockTextLLMProvider()
    if lname in {
        "aliyun",
        "aliyun-bailian",
        "bailian",
        "openai",
        "openai-compatible",
        "openai_compatible",
    }:
        return AliyunTextLLMProvider()
    return NullTextLLMProvider()


@lru_cache(maxsize=8)
def build_vlm_provider(name: str) -> VLMProvider:
    lname = (name or "none").lower().strip()
    if lname == "mock":
        return MockVLMProvider()
    if lname in {
        "aliyun",
        "aliyun-bailian",
        "bailian",
        "openai",
        "openai-compatible",
        "openai_compatible",
    }:
        return AliyunVLMProvider()
    return NullVLMProvider()
