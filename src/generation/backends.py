"""Generation backends behind one interface (proposal §5.2 models, ablation 5).

- hf_vlm: open-weights VLM via transformers (Qwen2.5-VL / Qwen3-VL class),
  wrapping src/generation/hf_vlm.py; bf16, batched, OOM fallback to singles.
- openai / gemini: API models over HTTPS (standard library only). Keys come
  from environment variables (OPENAI_API_KEY, GEMINI_API_KEY), never config
  files. Every attempt is checked against a persistent SpendLedger's cap
  before it is sent, using a conservative cost estimate. Per-token prices
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


def _is_finite_nonnegative(value: object) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value) and value >= 0
    except OverflowError:
        return False


class SpendLedger:
    """Cumulative API spend in a JSON file, shared by concurrent jobs via flock.

    reserve(estimate) refuses (SpendCapExceeded) if spent + reserved +
    estimate x ratio > cap, where ratio is the largest actual/estimate seen so
    far. Settlement replaces the reservation with the reported cost if known,
    or keeps the amount reserved when billing is uncertain. Provider costs
    are only known after a call and can exceed estimates.
    """

    def __init__(self, path: str | Path, cap_usd: float):
        if not _is_finite_nonnegative(cap_usd):
            raise ValueError("spend cap must be a finite non-negative number of USD")
        if fcntl is None:
            raise RuntimeError("API spend ledger requires file locking (fcntl)")
        self.path = Path(path)
        self.cap = float(cap_usd)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextlib.contextmanager
    def _locked(self):
        lock = self.path.with_suffix(self.path.suffix + ".lock")
        with open(lock, "a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                state = {"spent_usd": 0.0, "reserved_usd": 0.0, "calls": 0, "max_ratio": 1.0}
                if self.path.exists():
                    saved = json.loads(self.path.read_text(encoding="utf-8"))
                    if not isinstance(saved, dict) or any(
                        not _is_finite_nonnegative(saved.get(key)) for key in state
                    ):
                        raise ValueError(f"invalid API spend ledger: {self.path}")
                    state.update(saved)
                yield state
                tmp = self.path.with_suffix(self.path.suffix + ".tmp")
                tmp.write_text(json.dumps(state, indent=2, allow_nan=False), encoding="utf-8")
                tmp.replace(self.path)
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def reserve(self, estimate: float) -> float:
        """Reserve a scaled estimate; returns the amount reserved (pass it to settle)."""
        if not _is_finite_nonnegative(estimate) or estimate == 0:
            raise SpendCapExceeded("API attempt requires a finite positive cost estimate")
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
        if not isinstance(prices, dict):
            raise TypeError("prices must be a mapping")
        input_rate = prices["input_per_million"]
        output_rate = prices["output_per_million"]
        image_tokens = prices.get("image_tokens", 1500)
        # TOML environment interpolation can supply numbers as strings.
        if isinstance(input_rate, str):
            input_rate = float(input_rate)
        if isinstance(output_rate, str):
            output_rate = float(output_rate)
        if isinstance(image_tokens, str):
            image_tokens = int(image_tokens)
        if not (_is_finite_nonnegative(input_rate) and _is_finite_nonnegative(output_rate)):
            raise ValueError("token rates must be finite and non-negative")
        if input_rate == output_rate == 0:
            raise ValueError("at least one token rate must be positive")
        if not _is_finite_nonnegative(image_tokens) or image_tokens == 0 or int(image_tokens) != image_tokens:
            raise ValueError("image_tokens must be a positive integer")
        return Prices(float(input_rate), float(output_rate), int(image_tokens))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "API backends need prices.input_per_million and prices.output_per_million in the config "
            f"(check the provider's current price list); invalid cost model: {exc}"
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
    return SpendLedger(path, float(cap) if isinstance(cap, str) else cap)


def _post_json(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={
        "Content-Type": "application/json", **headers}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


class _UnreadableFrameError(OSError):
    pass


def _image_b64(path: str) -> tuple[str, str]:
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise _UnreadableFrameError(str(exc)) from exc
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    return mime, base64.b64encode(data).decode()


def _usage_count(usage: object, key: str) -> int | None:
    value = usage.get(key) if isinstance(usage, dict) else None
    return value if type(value) is int and _is_finite_nonnegative(value) else None


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
        if type(request.max_new_tokens) is not int or request.max_new_tokens <= 0:
            return GenResult(request.id, error="request: max_new_tokens must be a positive integer")
        try:
            estimate = self.prices.estimate(request.prompt, len(request.images), request.max_new_tokens)
        except OverflowError:
            return GenResult(request.id, error="spend_cap: request cost estimate overflowed")
        started, total_cost = time.time(), 0.0
        result = GenResult(request.id, error="request: retries exhausted")
        for attempt in range(self.max_retries + 1):
            try:
                reserved = self.ledger.reserve(estimate)
            except SpendCapExceeded as exc:
                result = GenResult(request.id, error=f"spend_cap: {exc}")
                break
            # An unreadable reply can still be billable. Keep this reservation unless
            # usage is known or the attempt was definitively rejected before generation.
            attempt_cost, calibrated_estimate = reserved, None
            retry = False
            try:
                text, raw, in_tok, out_tok, finished = self._call(request)
                reported_cost = self.prices.actual(in_tok or 0, out_tok or 0)
                full_usage = in_tok is not None and out_tok is not None
                if math.isfinite(reported_cost) and (full_usage or reported_cost > reserved):
                    attempt_cost, calibrated_estimate = reported_cost, estimate
                result = GenResult(request.id, text=text.strip(), raw=raw, new_tokens=out_tok,
                                   hit_token_limit=not finished,
                                   error=None if finished else f"hit_token_limit: max_new_tokens={request.max_new_tokens}")
            except urllib.error.HTTPError as exc:  # before OSError: HTTPError is a subclass of it
                if exc.code in (400, 401, 403, 404, 413, 422, 429):
                    attempt_cost = 0.0
                retry = exc.code in (429, 500, 502, 503)
                result = GenResult(request.id, error=f"http {exc.code}: {exc.reason}")
            except _UnreadableFrameError as exc:
                attempt_cost = 0.0
                result = GenResult(request.id, error=f"request: frame unreadable: {exc}")
            except (http.client.HTTPException, OSError, KeyError, IndexError, ValueError, TypeError) as exc:
                retry = True
                result = GenResult(request.id, error=f"request: {exc!r}")
            except Exception as exc:  # isolate an unexpected failure to this request
                result = GenResult(request.id, error=f"request: {exc!r}")
            finally:
                self.ledger.settle(reserved, attempt_cost, calibrated_estimate)
                total_cost += attempt_cost
            if not retry or attempt == self.max_retries:
                break
            time.sleep(min(30, 2 ** attempt))
        result.cost_usd = round(total_cost, 6)
        result.latency_s = round(time.time() - started, 3)
        return result


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
        return (choice["message"]["content"] or "", json.dumps(choice), _usage_count(usage, "prompt_tokens"),
                _usage_count(usage, "completion_tokens"), choice.get("finish_reason") != "length")


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
        return (text, json.dumps(candidate), _usage_count(usage, "promptTokenCount"),
                _usage_count(usage, "candidatesTokenCount"), candidate.get("finishReason") != "MAX_TOKENS")


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
        import torch
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

        self.config = dict(config or {})
        self.model_id = model
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model)
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model, torch_dtype=torch.bfloat16, device_map="auto").eval()
        if hasattr(self.model, "disable_talker"):
            self.model.disable_talker()

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
                conversation = [{"role": "user", "content": content}]
                text_in = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
                inputs = self.processor(text=text_in, images=[c["image"] for c in content if c["type"] == "image"] or None,
                                        audio=[audio] if audio is not None else None, return_tensors="pt", padding=True)
                inputs = inputs.to(self.model.device)
                started = time.time()
                with torch.inference_mode():
                    ids = self.model.generate(**inputs, max_new_tokens=r.max_new_tokens, do_sample=False, return_audio=False)
                new = ids[:, inputs["input_ids"].shape[-1]:]
                text = self.processor.batch_decode(new, skip_special_tokens=True)[0]
                n = int(new.shape[-1])
                out.append(GenResult(r.id, text.strip(), text, n, latency_s=round(time.time() - started, 3),
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
