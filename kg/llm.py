"""LLM backends used for triple extraction (operator L, Sec. 4.2) and for the
LLM-based reasoning baselines requested in review (R1-A, R2-6).

Backends
  * ClaudeBackend  - Anthropic API through the official `anthropic` SDK.
  * OllamaBackend  - local open-weight models (e.g. qwen2.5:7b-instruct,
                     llama3.1:8b-instruct) served by Ollama's native REST API.
Every call returns parsed JSON plus usage (tokens, latency) so the cost and
latency of the LLM component can be reported (R1-B). Responses are cached on
disk, keyed by backend, model and prompt, so re-running an experiment does
not call the API again.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from config import PROMPTS_DIR, RESULTS_DIR


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


class LLMError(RuntimeError):
    pass


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    cached_calls: int = 0

    def add(self, other: dict):
        self.calls += 1
        self.input_tokens += int(other.get("input_tokens", 0))
        self.output_tokens += int(other.get("output_tokens", 0))
        self.latency_s += float(other.get("latency_s", 0.0))


@dataclass
class LLMBackend:
    model: str
    cache_path: Path | None = None
    usage: Usage = field(default_factory=Usage)
    name: str = "base"

    def _raw(self, system: str, user: str, schema: dict) -> tuple[str, dict]:
        raise NotImplementedError

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        key = hashlib.sha256(json.dumps([self.name, self.model, system, user, schema]).encode()).hexdigest()
        cache = self._load_cache()
        if key in cache:
            self.usage.cached_calls += 1
            return cache[key]
        start = time.perf_counter()
        text, usage = self._raw(system, user, schema)
        usage["latency_s"] = time.perf_counter() - start
        self.usage.add(usage)
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError(f"{self.name} returned invalid JSON: {text[:200]}") from exc
        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "value": data}) + "\n")
            self._cache[key] = data
        return data

    def _load_cache(self) -> dict:
        if not hasattr(self, "_cache"):
            self._cache = {}
            if self.cache_path is not None and self.cache_path.exists():
                for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        row = json.loads(line)
                        self._cache[row["key"]] = row["value"]
        return self._cache

    def describe(self) -> dict:
        return {"backend": self.name, "model": self.model}


@dataclass
class ClaudeBackend(LLMBackend):
    model: str = "claude-opus-5-5"
    effort: str = "low"            # short classification/extraction answers
    max_tokens: int = 16000
    name: str = "anthropic"

    def __post_init__(self):
        try:
            import anthropic
        except ImportError as exc:
            raise LLMError("pip install anthropic  (and set ANTHROPIC_API_KEY)") from exc
        self._anthropic = anthropic
        self._client = anthropic.Anthropic()

    def _raw(self, system, user, schema):
        anthropic = self._anthropic
        try:
            response = self._client.beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": self.effort,
                               "format": {"type": "json_schema", "schema": schema}},
            )
        except anthropic.RateLimitError as exc:
            raise LLMError(f"rate limited: {exc.message}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("network error contacting the Anthropic API") from exc
        if response.stop_reason == "refusal":
            raise LLMError(f"request declined: {response.stop_details}")
        text = next(b.text for b in response.content if b.type == "text")
        return text, {"input_tokens": response.usage.input_tokens,
                      "output_tokens": response.usage.output_tokens}

    def describe(self):
        return {"backend": self.name, "model": self.model, "effort": self.effort}


@dataclass
class OllamaBackend(LLMBackend):
    model: str = "qwen2.5:7b-instruct"
    host: str = "http://localhost:11434"
    name: str = "ollama"

    def _raw(self, system, user, schema):
        body = json.dumps({
            "model": self.model,
            "stream": False,
            "format": schema,
            "options": {"temperature": 0, "seed": 0},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }).encode()
        req = urllib.request.Request(f"{self.host}/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = json.loads(resp.read())
        except OSError as exc:
            raise LLMError(f"cannot reach Ollama at {self.host}: {exc}") from exc
        return data["message"]["content"], {"input_tokens": data.get("prompt_eval_count", 0),
                                            "output_tokens": data.get("eval_count", 0)}

    def describe(self):
        return {"backend": self.name, "model": self.model, "temperature": 0}


@dataclass
class DeepSeekBackend(LLMBackend):
    """DeepSeek (or any OpenAI-compatible chat-completions endpoint).

    Key:      env DEEPSEEK_API_KEY   (never hard-code it)
    Endpoint: env DEEPSEEK_BASE_URL  (default https://api.deepseek.com; set it to
              your provider's URL if the key was issued by a third-party gateway)
    JSON mode is requested with response_format=json_object and the schema is
    given in the system prompt; the reply is validated by json.loads.
    """
    model: str = "deepseek-flash"
    base_url: str = ""
    name: str = "deepseek"
    max_tokens: int = 8000

    def __post_init__(self):
        import os
        self.api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not self.api_key:
            raise LLMError("set the DEEPSEEK_API_KEY environment variable")
        self.base_url = (self.base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
        # provider-specific request fields, e.g. '{"enable_thinking": false}'
        self.extra_body = json.loads(os.environ.get("DEEPSEEK_EXTRA_BODY", "") or "{}")
        self.json_mode = True          # switched off automatically if the endpoint rejects it

    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.api_key}"})
        last = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=600) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:300]
                if exc.code in (429, 500, 502, 503, 504) and attempt < 3:
                    last = exc
                    time.sleep(2 ** attempt * 5)
                    continue
                raise _HTTPError(exc.code, detail) from exc
            except OSError as exc:
                last = exc
                time.sleep(2 ** attempt * 5)
        raise LLMError(f"cannot reach {self.base_url}: {last}")

    def _raw(self, system, user, schema):
        system = (system + "\n\nReply with a single JSON object (no prose, no code fences) that conforms "
                  "to this JSON schema:\n" + json.dumps(schema))
        payload = {"model": self.model,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                   "temperature": 0, "max_tokens": self.max_tokens, "stream": False, **self.extra_body}
        try:
            data = self._post({**payload, "response_format": {"type": "json_object"}} if self.json_mode else payload)
        except _HTTPError as exc:
            if not (self.json_mode and exc.code in (400, 422)):
                raise LLMError(f"{self.name} HTTP {exc.code}: {exc.detail}") from exc
            self.json_mode = False     # endpoint does not support response_format
            try:
                data = self._post(payload)
            except _HTTPError as exc2:
                raise LLMError(f"{self.name} HTTP {exc2.code}: {exc2.detail}") from exc2
        choice = data["choices"][0]
        if choice.get("finish_reason") == "length":
            raise LLMError("reply truncated (max_tokens); use a smaller batch")
        usage = data.get("usage") or {}
        return _json_text(choice["message"]["content"] or ""), {
            "input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)}

    def describe(self):
        return {"backend": self.name, "model": self.model, "base_url": self.base_url, "temperature": 0,
                "json_mode": self.json_mode, "extra_body": self.extra_body}


class _HTTPError(Exception):
    def __init__(self, code, detail):
        super().__init__(f"HTTP {code}: {detail}")
        self.code, self.detail = code, detail


def _json_text(text: str) -> str:
    """Strip code fences / reasoning text around the outermost JSON object."""
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end > start else text


def safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "-" for c in s)


def make_backend(kind: str, model: str | None = None, tag: str = "llm") -> LLMBackend:
    cache = RESULTS_DIR / "llm_cache" / f"{tag}_{kind}_{safe_name(model or 'default')}.jsonl"
    if kind == "anthropic":
        return ClaudeBackend(model=model or "claude-opus-5-5", cache_path=cache)
    if kind == "ollama":
        return OllamaBackend(model=model or "qwen2.5:7b-instruct", cache_path=cache)
    if kind == "deepseek":
        return DeepSeekBackend(model=model or "deepseek-flash", cache_path=cache)
    raise ValueError(f"unknown LLM backend {kind!r} (use 'anthropic', 'deepseek' or 'ollama')")
