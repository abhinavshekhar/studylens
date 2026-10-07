"""Provider-agnostic LLM adapter for StudyLens."""

from __future__ import annotations

import importlib
import json
import time
import urllib.error
import urllib.request
from typing import Callable, Sequence

import config


class LLMError(RuntimeError):
    """Raised when text generation fails after retries."""


Message = dict[str, str]


def generate(system: str, messages: Sequence[Message], max_tokens: int = 512) -> str:
    """Generate a response using the configured provider and model."""
    importlib.reload(config)
    config.refresh_settings()
    provider = config.LLM_PROVIDER.strip().lower()
    model = config.LLM_MODEL.strip()

    if provider == "gemini":
        return _retry_once(lambda: _generate_gemini(system, messages, model, max_tokens))
    if provider == "ollama":
        return _retry_once(lambda: _generate_ollama(system, messages, model, max_tokens))
    if provider == "groq":
        return _retry_once(lambda: _generate_groq(system, messages, model, max_tokens))

    raise LLMError(
        f"Unsupported LLM provider '{config.LLM_PROVIDER}'. Supported providers: gemini, ollama, groq."
    )


def _retry_once(call: Callable[[], str]) -> str:
    """Retry one transient failure before surfacing a user-friendly error."""
    try:
        return call()
    except Exception as exc:  # pragma: no cover - behavior tested through public API mocks.
        if not _is_transient_error(exc):
            raise _friendly_error(exc) from exc
        time.sleep(1.0)
        try:
            return call()
        except Exception as retry_exc:
            raise _friendly_error(retry_exc) from retry_exc


def _friendly_error(exc: Exception) -> LLMError:
    text = str(exc).strip() or exc.__class__.__name__
    return LLMError(f"The language model request failed. Please retry. Details: {text}")


def _is_transient_error(exc: Exception) -> bool:
    message = str(exc).lower()
    if any(token in message for token in ("rate", "timeout", "tempor", "unavailable", "overloaded")):
        return True
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in {408, 409, 425, 429, 500, 502, 503, 504}
    return False


def _generate_gemini(system: str, messages: Sequence[Message], model: str, max_tokens: int) -> str:
    api_key = config.GEMINI_API_KEY.strip()
    if not api_key:
        raise LLMError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini.")

    try:
        from google import genai
        from google.genai import types as genai_types
    except Exception as exc:  # pragma: no cover
        raise LLMError(
            "Gemini SDK not available. Install dependencies from requirements.txt."
        ) from exc

    client = genai.Client(api_key=api_key)
    contents = [{"role": msg["role"], "parts": [{"text": msg["content"]}]} for msg in messages]
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=genai_types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            temperature=0.2,
        ),
    )

    text = getattr(response, "text", "")
    if text:
        return text.strip()
    raise LLMError("Gemini returned an empty response.")


def _generate_ollama(system: str, messages: Sequence[Message], model: str, max_tokens: int) -> str:
    url = config.OLLAMA_BASE_URL.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "messages": [{"role": "system", "content": system}, *messages],
        "options": {"num_predict": max_tokens, "temperature": 0.2},
    }
    data = _http_json("POST", url, payload)
    content = (((data.get("message") or {}).get("content")) or "").strip()
    if not content:
        raise LLMError("Ollama returned an empty response.")
    return content


def _generate_groq(system: str, messages: Sequence[Message], model: str, max_tokens: int) -> str:
    api_key = config.GROQ_API_KEY.strip()
    if not api_key:
        raise LLMError("GROQ_API_KEY is required when LLM_PROVIDER=groq.")

    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, *messages],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }
    data = _http_json(
        "POST",
        "https://api.groq.com/openai/v1/chat/completions",
        payload,
        headers={"Authorization": "Bearer " + api_key},
    )

    choices = data.get("choices") or []
    if not choices:
        raise LLMError("Groq returned no choices.")
    content = (((choices[0].get("message") or {}).get("content")) or "").strip()
    if not content:
        raise LLMError("Groq returned an empty response.")
    return content


def _http_json(method: str, url: str, payload: dict, headers: dict[str, str] | None = None) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req_headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if headers:
        req_headers.update(headers)
    request = urllib.request.Request(url, data=body, method=method, headers=req_headers)

    with urllib.request.urlopen(request, timeout=45) as response:
        data = response.read().decode("utf-8")
    return json.loads(data)
