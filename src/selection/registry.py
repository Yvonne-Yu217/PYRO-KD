"""One interface over every selector and ablation variant.

A selector spec is {"name": str, "kind": one of KINDS, "options": {...},
"budget": "flat" | "hierarchical", "hierarchy": {...}}. `name` is the label in
outputs; `kind` picks the implementation:

- U: uniform t_i = i * D / (K + 1) (src/utils/selection.py)
- O: oracle, human AD starts capped at K (upper bound; never deployable)
- A: the team's gap-quality audio selector (src/utils/audio_selector.py),
     unchanged; needs VAD features
- V: the team's CLIP-novelty + shot selector (src/utils/visual_features.py),
     unchanged; needs visual features
- two_stage: the proposal's fused selector (src/selection/two_stage.py); AV,
     A_cue and V_mem are presets of it (see PRESETS)

Every selector receives the same K and returns times in seconds; nothing here
generates text, so selector-only runs never touch an MLLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

from src.selection import two_stage
from src.selection.cues import CueBundle, build_cue_bundle
from src.selection.hierarchical import select_hierarchical
from src.utils.audio_selector import AudioSelector
from src.utils.selection import oracle_timestamps, uniform_timestamps

KINDS = ("U", "O", "A", "V", "two_stage")

# Presets of the two-stage selector. Weights are untuned placeholders.
PRESETS: dict[str, dict[str, Any]] = {
    # AV: the full fused score with running novelty memory.
    "AV": {"novelty_mode": "memory"},
    # A as the proposal defines it (speech cues + NSE, snapped to VAD gaps, no
    # visual input). Coexists with the team's A; see DEVELOPMENT_LOG.
    "A_cue": {
        "weights": {"vn": 0.0, "av_disagreement": 0.0},
        "candidate_streams": ["nse", "sc"],
        "diversity": "mmr",
    },
    # V with running-memory novelty (ablation 8), placed at the event time like V.
    "V_mem": {
        "weights": {"nse": 0.0, "sc": 0.0, "av_disagreement": 0.0, "gf": 0.0, "op": 0.0},
        "candidate_streams": ["vn"],
        "novelty_mode": "memory",
        "placement": "event",
        "require_vad": False,
    },
}

DEFAULT_SPECS: dict[str, dict[str, Any]] = {
    "U": {"name": "U", "kind": "U"},
    "O": {"name": "O", "kind": "O"},
    "A": {"name": "A", "kind": "A"},
    "V": {"name": "V", "kind": "V"},
    "AV": {"name": "AV", "kind": "two_stage", "options": PRESETS["AV"]},
    "A_cue": {"name": "A_cue", "kind": "two_stage", "options": PRESETS["A_cue"]},
    "V_mem": {"name": "V_mem", "kind": "two_stage", "options": PRESETS["V_mem"]},
}


@dataclass
class SelectionContext:
    """Validated inputs for one video; any field may be None when unavailable."""

    video_id: str
    duration: float
    audio_features: dict | None = None
    audio_cues: dict | None = None
    visual_features: dict | None = None
    embeddings: Any = None
    ground_truth: list[float] | None = None  # O only
    missing: dict[str, str] = field(default_factory=dict)
    bundle_config: dict | None = None
    _bundles: dict = field(default_factory=dict)

    def bundle(self, use_visual: bool = True) -> CueBundle:
        """The cue bundle; use_visual=False builds it from audio inputs only, so a selector that is
        defined as audio-only cannot pick up CLIP novelty, shots or embeddings from a visual cache."""
        if use_visual not in self._bundles:
            self._bundles[use_visual] = build_cue_bundle(
                self.duration, self.audio_features, self.audio_cues,
                self.visual_features if use_visual else None, self.embeddings if use_visual else None,
                self.bundle_config, self.missing,
            )
        return self._bundles[use_visual]


@dataclass
class Selection:
    selector: str
    times: list[float]
    status: str
    k: int
    details: list[dict] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    @property
    def shortfall(self) -> int:
        return self.k - len(self.times)


def resolve_spec(spec: str | dict) -> dict:
    """A spec dict from a name (DEFAULT_SPECS) or a dict, with options merged onto its preset."""
    if isinstance(spec, str):
        if spec not in DEFAULT_SPECS:
            raise KeyError(f"unknown selector {spec!r}; known: {sorted(DEFAULT_SPECS)}")
        return json.loads(json.dumps(DEFAULT_SPECS[spec]))
    spec = json.loads(json.dumps(spec))
    base = DEFAULT_SPECS.get(spec.get("base", ""), {})
    merged = {**base, **{k: v for k, v in spec.items() if k != "base"}}
    if "base" in spec and base.get("kind") == "two_stage":
        options = json.loads(json.dumps(base.get("options", {})))
        for key, value in spec.get("options", {}).items():
            if isinstance(value, dict) and isinstance(options.get(key), dict):
                options[key].update(value)
            else:
                options[key] = value
        merged["options"] = options
    if merged.get("kind") not in KINDS:
        raise ValueError(f"selector kind must be one of {KINDS}; got {merged.get('kind')!r}")
    merged.setdefault("name", merged["kind"])
    if merged["kind"] == "two_stage":
        two_stage.merge_config(merged.get("options"))  # validate early
    return merged


def spec_uses_visual(spec: str | dict) -> bool:
    """True if the selector reads visual features: V, or a two-stage spec with a visual score term."""
    spec = resolve_spec(spec)
    if spec["kind"] == "V":
        return True
    if spec["kind"] != "two_stage":
        return False
    weights = two_stage.merge_config(spec.get("options"))["weights"]
    return weights.get("vn", 0) != 0 or weights.get("av_disagreement", 0) != 0


def run_selector(spec: str | dict, context: SelectionContext, k: int) -> Selection:
    """Select up to K moments for one video. Missing inputs give an explicit status."""
    spec = resolve_spec(spec)
    name, kind = spec["name"], spec["kind"]
    if kind == "U":
        return Selection(name, uniform_timestamps(context.duration, k), "ok", k)
    if kind == "O":
        if context.ground_truth is None:
            return Selection(name, [], "missing_ground_truth", k)
        return Selection(name, oracle_timestamps(context.ground_truth, k), "ok", k)
    if kind == "A":
        if context.audio_features is None:
            return Selection(name, [], "missing_audio_features", k,
                             diagnostics={"reason": context.missing.get("vad")})
        picks = AudioSelector(**spec.get("options", {})).select(context.audio_features, k)
        return Selection(name, [c.time for c in picks], "ok", k,
                         [{"time": c.time, "gap_start": c.gap_start, "gap_end": c.gap_end, "score": c.score} for c in picks])
    if kind == "V":
        if context.visual_features is None:
            return Selection(name, [], "missing_visual_features", k,
                             diagnostics={"reason": context.missing.get("visual")})
        from src.utils.visual_features import VisualCueSelector

        picks = VisualCueSelector(**spec.get("options", {})).select(context.visual_features, k)
        return Selection(name, [c.time for c in picks], "ok", k,
                         [{"time": c.time, "score": c.score, "novelty": c.novelty, "shot_boundary": c.shot_boundary} for c in picks])
    options = spec.get("options", {})
    cfg = two_stage.merge_config(options)
    uses_visual = spec_uses_visual(spec)
    if uses_visual and context.visual_features is None:
        return Selection(name, [], "missing_visual_features", k, diagnostics={"reason": context.missing.get("visual")})
    bundle = context.bundle(use_visual=uses_visual)
    if spec.get("budget", "flat") == "hierarchical":
        result = select_hierarchical(bundle, k, options, spec.get("hierarchy"))
    else:
        result = two_stage.select(bundle, k, options)
    status = result.status
    if status == "ok" and context.audio_cues is None and (
        cfg["weights"].get("nse", 0) != 0 or cfg["candidate_streams"] != ["vn"]
    ):
        status = "ok_degraded"  # ran without the NSE/SC streams; see diagnostics
    return Selection(name, result.times, status, k, result.selected, result.diagnostics)
