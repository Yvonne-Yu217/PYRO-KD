"""Frozen VLM access through an OpenAI-compatible chat endpoint.

Both backbones are served the same way (vLLM on HPC: `vllm serve <model>`),
so every selector/prior condition shares one generation interface and the
model is swapped by name only. Uses only the standard library so it runs in
the main project env; vLLM lives in its own env (see scripts/hpc/).
"""

import base64
from dataclasses import dataclass, field
import io
import json
import mimetypes
from pathlib import Path
import time
from typing import Any, Sequence
import urllib.error
import urllib.request

QWEN3_VL_8B = "Qwen/Qwen3-VL-8B-Instruct"
MUSE_GLIMMER_30B = "nvidia/Muse-Glimmer-30B-NVFP4"


@dataclass(frozen=True)
class GenerationResult:
    """One generation plus the cost/latency fields the proposal asks us to log."""

    text: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_seconds: float
    request: dict[str, Any] = field(repr=False)
    raw_response: dict[str, Any] = field(repr=False)


def _image_data_url(image: Any) -> str:
    """Encode a file path or PIL image as a base64 data URL."""
    if isinstance(image, (str, Path)):
        path = Path(image)
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        payload = path.read_bytes()
    else:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=90)
        mime, payload = "image/jpeg", buffer.getvalue()
    return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"


class VLMClient:
    """Chat-completions client for a vLLM (or other OpenAI-compatible) server."""

    def __init__(
        self,
        model: str = QWEN3_VL_8B,
        base_url: str = "http://localhost:8000/v1",
        api_key: str | None = None,
        timeout_seconds: float = 300.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError(f"VLM server returned {exc.code}: {detail}") from exc

    def build_request(
        self,
        images: Sequence[Any],
        prompt: str,
        system: str | None = None,
        max_tokens: int = 128,
        temperature: float = 0.0,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Images come first, in chronological order, then the text prompt."""
        content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": _image_data_url(image)}}
            for image in images
        ]
        content.append({"type": "text", "text": prompt})
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})
        return {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
        }

    def generate(
        self,
        images: Sequence[Any],
        prompt: str,
        system: str | None = None,
        max_tokens: int = 128,
        temperature: float = 0.0,
        seed: int = 0,
    ) -> GenerationResult:
        request = self.build_request(images, prompt, system, max_tokens, temperature, seed)
        started = time.perf_counter()
        response = self._post("/chat/completions", request)
        latency = time.perf_counter() - started
        usage = response.get("usage") or {}
        return GenerationResult(
            text=response["choices"][0]["message"]["content"].strip(),
            model=response.get("model", self.model),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            latency_seconds=latency,
            request={k: v for k, v in request.items() if k != "messages"} | {
                "num_images": len(images), "prompt": prompt, "system": system,
            },
            raw_response=response,
        )
