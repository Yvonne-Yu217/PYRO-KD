"""Generation backends behind one interface (proposal §5.2 models, ablation 5).

- hf_vlm: open-weights VLM via transformers (Qwen2.5-VL / Qwen3-VL class),
  wrapping src/generation/hf_vlm.py; bf16, batched, OOM fallback to singles.
- openai / gemini: API models over HTTPS (standard library only). Keys come
  from environment variables (OPENAI_API_KEY, GEMINI_API_KEY), never config
  files. Every call is checked against a persistent SpendLedger with a hard
  cap *before* it is sent, using a worst-case cost estimate; per-token prices
  must be set in the config (no built-in prices, since they change).
- qwen_omni: audio-native Qwen2.5-Omni via transformers (ablation 5): hears the
  moment's audio clip as well as the frames.
- echo: deterministic stand-in that returns a fixed string. It exists for
  dry runs and pipeline tests and is never a model result; outputs record
  backend="echo".

Temperature is 0 everywhere (held-constant control); a backend that cannot
honour that must refuse. None of the real backends has been run from this
laptop (no GPU, no transformers, no API keys).
"""

from __future__ import annotations

import base64
import contextlib
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Protocol
import http.client
import urllib.error
import urllib.request

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


@dataclass
class GenRequest:
    id: str
    prompt: str
    images: list[str] = field(default_factory=list)  # frame paths, chronological
    audio: dict | None = None  # {"path", "start", "end"} for audio-native models
    max_new_tokens: int = 64


@dataclass
class GenResult:
    id: str
    text: str = ""
    raw: str = ""
    new_tokens: int | None = None
    error: str | None = None
    cost_usd: float = 0.0
    latency_s: float = 0.0
    hit_token_limit: bool = False

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if v is not None}
        if self.error is None:
            d.pop("error", None)
        return d


class Backend(Protocol):
    name: str
    model_id: str
    supports_audio: bool

    def generate(self, requests: list[GenRequest]) -> list[GenResult]: ...


class SpendCapExceeded(RuntimeError):
    pass


class SpendLedger:
    """Cumulative API spend in a JSON file, shared by concurrent jobs via flock.

    reserve(estimate) refuses (SpendCapExceeded) if spent + reserved +
    estimate x ratio > cap, where ratio is the largest actual/estimate seen so
    far (token counts are only known after a call). settle(estimate, actual)
    replaces the reservation with the actual cost and updates the ratio. The
    cap can therefore be exceeded by at most one call's underestimate, before
    the ratio has been learned; keep a small margin below the real budget.
    """

    def __init__(self, path: str | Path, cap_usd: float):
        if not (isinstance(cap_usd, (int, float)) and math.isfinite(cap_usd) and cap_usd >= 0):
            raise ValueError("spend cap must be a finite non-negative number of USD")
        self.path = Path(path)
        self.cap = float(cap_usd)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextlib.contextmanager
    def _locked(self):
        lock = self.path.with_suffix(self.path.suffix + ".lock")
        with open(lock, "a+") as handle:
            if fcntl is not None:
                fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                state = {"spent_usd": 0.0, "reserved_usd": 0.0, "calls": 0, "max_ratio": 1.0}
                if self.path.exists():
                    state.update(json.loads(self.path.read_text(encoding="utf-8") or "{}"))
                yield state
                tmp = self.path.with_suffix(".tmp")
                tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
                tmp.replace(self.path)
            finally:
                if fcntl is not None:
                    fcntl.flock(handle, fcntl.LOCK_UN)

    def reserve(self, estimate: float) -> float:
        """Reserve a scaled estimate; returns the amount reserved (pass it to settle)."""
        with self._locked() as state:
            scaled = estimate * max(1.0, state.get("max_ratio", 1.0))
            committed = state["spent_usd"] + state["reserved_usd"]
            if committed + scaled > self.cap:
                raise SpendCapExceeded(
                    f"API spend cap ${self.cap:.2f} would be exceeded "
                    f"(committed ${committed:.4f} + estimate ${scaled:.4f}); raise the cap deliberately to continue"
                )
            state["reserved_usd"] += scaled
            return scaled

    def settle(self, reserved: float, actual: float, estimate: float | None = None) -> None:
        with self._locked() as state:
            state["reserved_usd"] = max(0.0, state["reserved_usd"] - reserved)
            state["spent_usd"] += actual
            state["calls"] += 1
            if estimate and actual > 0:
                state["max_ratio"] = max(state.get("max_ratio", 1.0), actual / estimate)

    def spent(self) -> float:
        with self._locked() as state:
            return state["spent_usd"]


@dataclass
class Prices:
    """USD per 1M tokens; image_tokens is a conservative per-image input estimate."""

    input_per_million: float
    output_per_million: float
    image_tokens: int = 1500

    def estimate(self, prompt: str, n_images: int, max_new_tokens: int) -> float:
        input_tokens = len(prompt) / 3.0 + n_images * self.image_tokens + 50
        return (input_tokens * self.input_per_million + max_new_tokens * self.output_per_million) / 1e6

    def actual(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_per_million + output_tokens * self.output_per_million) / 1e6


def _require_prices(config: dict) -> Prices:
    prices = config.get("prices") or {}
    try:
        return Prices(float(prices["input_per_million"]), float(prices["output_per_million"]),
                      int(prices.get("image_tokens", 1500)))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "API backends need prices.input_per_million and prices.output_per_million in the config "
            "(check the provider's current price list); refusing to run without a cost model"
        ) from exc


def _ledger(config: dict) -> SpendLedger:
    cap = config.get("spend_cap_usd", os.environ.get("PYRO_API_SPEND_CAP_USD"))
    if cap is None:
        raise ValueError("API backends need spend_cap_usd in the config or PYRO_API_SPEND_CAP_USD; refusing without a cap")
    path = config.get("ledger_path") or os.environ.get("PYRO_API_LEDGER")
    if not path or not Path(path).is_absolute():
        # A relative path would resolve against whatever directory the job started in, giving each
        # job its own ledger at $0 and defeating the shared cap. run_experiment supplies a per-experiment path.
        raise ValueError(
            "API backends need an absolute ledger_path (config) or PYRO_API_LEDGER so every job shares one spend ledger"
        )
    return SpendLedger(path, float(cap))


def _post_json(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={
        "Content-Type": "application/json", **headers}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def _image_b64(path: str) -> tuple[str, str]:
    data = Path(path).read_bytes()
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    return mime, base64.b64encode(data).decode()


class _APIBackend:
    supports_audio = False
    key_env = ""

    def __init__(self, model_id: str, config: dict | None = None, transport=None):
        self.config = dict(config or {})
        if float(self.config.get("temperature", 0.0)) != 0.0:
            raise ValueError("temperature is held at 0.0 for every configuration")
        self.model_id = model_id
        self.prices = _require_prices(self.config)
        self.ledger = _ledger(self.config)
        self.api_key = os.environ.get(self.config.get("api_key_env", self.key_env))
        if not self.api_key:
            raise ValueError(f"set {self.config.get('api_key_env', self.key_env)} in the environment")
        self.timeout = float(self.config.get("timeout_s", 60))
        self.max_retries = int(self.config.get("max_retries", 3))
        self.transport = transport or _post_json

    def generate(self, requests: list[GenRequest]) -> list[GenResult]:
        return [self._one(r) for r in requests]

    def _one(self, request: GenRequest) -> GenResult:
        estimate = self.prices.estimate(request.prompt, len(request.images), request.max_new_tokens)
        try:
            reserved = self.ledger.reserve(estimate)
        except SpendCapExceeded as exc:
            return GenResult(request.id, error=f"spend_cap: {exc}")
        started, actual, maybe_billed = time.time(), 0.0, False
        try:
            for attempt in range(self.max_retries + 1):
                try:
                    text, raw, in_tok, out_tok, finished = self._call(request)
                    actual = self.prices.actual(in_tok, out_tok)
                    return GenResult(request.id, text=text.strip(), raw=raw, new_tokens=out_tok, cost_usd=round(actual, 6),
                                     latency_s=round(time.time() - started, 3), hit_token_limit=not finished,
                                     error=None if finished else f"hit_token_limit: max_new_tokens={request.max_new_tokens}")
                except urllib.error.HTTPError as exc:  # before URLError: HTTPError is a subclass of it
                    if exc.code in (429, 500, 502, 503) and attempt < self.max_retries:
                        time.sleep(min(30, 2 ** attempt))
                        continue
                    return GenResult(request.id, error=f"http {exc.code}: {exc.reason}")
                except FileNotFoundError as exc:  # a frame file is gone: retrying cannot help, nothing was sent
                    return GenResult(request.id, error=f"request: frame unreadable: {exc!r}")
                except (TimeoutError, ConnectionError, http.client.HTTPException, json.JSONDecodeError) as exc:
                    maybe_billed = True  # the request reached the provider; the reply was lost or unreadable
                    if attempt < self.max_retries:
                        time.sleep(min(30, 2 ** attempt))
                        continue
                    return GenResult(request.id, error=f"request: {exc!r}")
                except (urllib.error.URLError, OSError, KeyError, IndexError, ValueError, TypeError) as exc:
                    if attempt < self.max_retries:
                        time.sleep(min(30, 2 ** attempt))
                        continue
                    return GenResult(request.id, error=f"request: {exc!r}")
            return GenResult(request.id, error="request: retries exhausted")
        except Exception as exc:  # never let one request take down the batch
            return GenResult(request.id, error=f"request: {exc!r}")
        finally:
            # A reply that never arrived may still have been billed: charge the reserved worst case rather than $0.
            self.ledger.settle(reserved, actual if actual or not maybe_billed else reserved, estimate)


class OpenAIBackend(_APIBackend):
    name = "openai"
    key_env = "OPENAI_API_KEY"

    def _call(self, request: GenRequest):
        content: list[dict[str, Any]] = []
        for path in request.images:
            mime, data = _image_b64(path)
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}", "detail": "low"}})
        content.append({"type": "text", "text": request.prompt})
        payload = {"model": self.model_id, "messages": [{"role": "user", "content": content}],
                   "temperature": 0.0, "max_tokens": request.max_new_tokens, "seed": int(self.config.get("seed", 0))}
        url = self.config.get("base_url", "https://api.openai.com/v1") + "/chat/completions"
        reply = self.transport(url, payload, {"Authorization": f"Bearer {self.api_key}"}, self.timeout)
        choice = reply["choices"][0]
        usage = reply.get("usage", {})
        return (choice["message"]["content"] or "", json.dumps(choice), int(usage.get("prompt_tokens", 0)),
                int(usage.get("completion_tokens", 0)), choice.get("finish_reason") != "length")


class GeminiBackend(_APIBackend):
    name = "gemini"
    key_env = "GEMINI_API_KEY"

    def _call(self, request: GenRequest):
        parts: list[dict[str, Any]] = []
        for path in request.images:
            mime, data = _image_b64(path)
            parts.append({"inline_data": {"mime_type": mime, "data": data}})
        parts.append({"text": request.prompt})
        payload = {"contents": [{"role": "user", "parts": parts}],
                   "generationConfig": {"temperature": 0.0, "maxOutputTokens": request.max_new_tokens}}
        base = self.config.get("base_url", "https://generativelanguage.googleapis.com/v1beta")
        reply = self.transport(f"{base}/models/{self.model_id}:generateContent", payload,
                               {"x-goog-api-key": self.api_key}, self.timeout)
        candidate = reply["candidates"][0]
        text = "".join(p.get("text", "") for p in candidate.get("content", {}).get("parts", []))
        usage = reply.get("usageMetadata", {})
        return (text, json.dumps(candidate), int(usage.get("promptTokenCount", 0)),
                int(usage.get("candidatesTokenCount", 0)), candidate.get("finishReason") != "MAX_TOKENS")


class HFVLMBackend:
    """Open-weights VLM through transformers (bf16, device_map=auto)."""

    name = "hf_vlm"
    supports_audio = False

    def __init__(self, model: str, config: dict | None = None):
        from src.generation.hf_vlm import load_model, resolve_model

        self.config = dict(config or {})
        if float(self.config.get("temperature", 0.0)) != 0.0:
            raise ValueError("temperature is held at 0.0 for every configuration")
        self.model_id = resolve_model(model)
        self.processor, self.model, self.model_class = load_model(
            self.model_id, self.config.get("attn_impl"), self.config.get("quant", "none"))
        self.batch_size = int(self.config.get("batch_size", 4))
        self.max_side = self.config.get("max_side", 768)

    def generate(self, requests: list[GenRequest]) -> list[GenResult]:
        import torch
        from PIL import Image

        from src.generation.hf_vlm import generate_batch

        def load(r):
            images = []
            for p in r.images:
                image = Image.open(p).convert("RGB")
                if self.max_side and max(image.size) > self.max_side:
                    scale = self.max_side / max(image.size)
                    image = image.resize((round(image.width * scale), round(image.height * scale)), Image.BICUBIC)
                images.append(image)
            return images

        results: dict[str, GenResult] = {}
        pending = [requests[i:i + self.batch_size] for i in range(0, len(requests), self.batch_size)]
        while pending:
            batch = pending.pop(0)
            try:
                started = time.time()
                texts, raws, n_new = generate_batch(
                    self.processor, self.model, [load(r) for r in batch], [r.prompt for r in batch],
                    max_new_tokens=max(r.max_new_tokens for r in batch), temperature=0.0,
                    reasoning_strength=self.config.get("reasoning_strength"),
                )
                dt = time.time() - started
                for r, text, raw, n in zip(batch, texts, raws, n_new):
                    hit = n >= r.max_new_tokens
                    results[r.id] = GenResult(r.id, text, raw, n, latency_s=round(dt, 3), hit_token_limit=hit,
                                              error=f"hit_token_limit: max_new_tokens={r.max_new_tokens}" if hit else None)
            except Exception as exc:  # OOM etc.: retry one at a time, then record
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if len(batch) > 1:
                    pending = [[r] for r in batch] + pending
                else:
                    results[batch[0].id] = GenResult(batch[0].id, error=f"generate: {exc!r}")
        return [results[r.id] for r in requests]


class QwenOmniBackend:
    """Audio-native ablation (Qwen2.5-Omni): frames plus the moment's audio clip.

    Follows the transformers Qwen2.5-Omni interface (Qwen2_5OmniForConditionalGeneration,
    text output only). Untested here; verify on HPC with a 4-item smoke test first.
    """

    name = "qwen_omni"
    supports_audio = True

    def __init__(self, model: str = "Qwen/Qwen2.5-Omni-7B", config: dict | None = None):
        from src.generation.omni import load_model

        self.config = dict(config or {})
        self.model_id = model
        self.processor, self.model, self.model_class = load_model(model, self.config.get("attn_impl"))

    def generate(self, requests: list[GenRequest]) -> list[GenResult]:
        import torch
        from PIL import Image

        from src.utils.media import load_audio

        out = []
        for r in requests:
            try:
                content: list[dict[str, Any]] = [{"type": "image", "image": Image.open(p).convert("RGB")} for p in r.images]
                audio = None
                if r.audio:
                    wave, sr = load_audio(r.audio["path"], 16_000)
                    audio = wave[int(r.audio["start"] * sr):int(r.audio["end"] * sr)]
                    content.append({"type": "audio", "audio": audio})
                content.append({"type": "text", "text": r.prompt})
                from src.generation.omni import generate_one
                started = time.time()
                text, raw, counts = generate_one(
                    self.processor, self.model,
                    [c["image"] for c in content if c["type"] == "image"], r.prompt,
                    waveform=audio, max_new_tokens=r.max_new_tokens, temperature=0.0,
                )
                n = counts[0]
                out.append(GenResult(r.id, text[0], raw[0], n, latency_s=round(time.time() - started, 3),
                                     hit_token_limit=n >= r.max_new_tokens,
                                     error=f"hit_token_limit: max_new_tokens={r.max_new_tokens}" if n >= r.max_new_tokens else None))
            except Exception as exc:
                out.append(GenResult(r.id, error=f"generate: {exc!r}"))
        return out


class EchoBackend:
    """Deterministic stand-in for dry runs and tests; never a model result."""

    name = "echo"
    supports_audio = True

    def __init__(self, model: str = "echo", config: dict | None = None):
        self.model_id = model
        self.reply = (config or {}).get("reply", "A person stands in a room.")

    def generate(self, requests: list[GenRequest]) -> list[GenResult]:
        return [GenResult(r.id, self.reply, self.reply, len(self.reply.split())) for r in requests]


def build_backend(spec: dict) -> Backend:
    """spec: {"backend": "hf_vlm" | "openai" | "gemini" | "qwen_omni" | "echo", "model": ..., ...}."""
    kind = spec.get("backend")
    model = spec.get("model")
    if kind == "hf_vlm":
        return HFVLMBackend(model, spec)
    if kind == "openai":
        return OpenAIBackend(model, spec)
    if kind == "gemini":
        return GeminiBackend(model, spec)
    if kind == "qwen_omni":
        return QwenOmniBackend(model or "Qwen/Qwen2.5-Omni-7B", spec)
    if kind == "echo":
        return EchoBackend(model or "echo", spec)
    raise ValueError(f"unknown backend {kind!r}")
