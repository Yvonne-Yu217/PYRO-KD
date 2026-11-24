"""The core 5 x 2 x 2 matrix and the 8 ablations (proposal §5.2-§5.3) as run lists.

A selection run is (selector spec) and is shared by every prior/model that
uses it, so selections are computed once. A generation run is (selector,
prior, model, timeline). Run IDs are readable labels plus a hash of
everything that affects the result, so a changed setting gets a new ID and
never silently reuses old outputs.

The A slot of the matrix is configurable (matrix.a_selector): the team's
gap-quality "A" by default, or the proposal-style "A_cue". O is always
flagged as an oracle upper bound.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

from src.experiments.config import ConfigError, get, stable_hash
from src.selection.registry import resolve_spec

ORACLE_KINDS = {"O"}

# Model-spec keys that change cost, throughput or credentials but not what a model returns. They are
# left out of run_id so raising the spend cap or changing a batch size resumes the same run instead of
# regenerating (and re-paying for) every moment under a new id.
NON_RESULT_MODEL_KEYS = frozenset({
    "spend_cap_usd", "ledger_path", "prices", "timeout_s", "max_retries", "request_batch", "batch_size",
})


@dataclass(frozen=True)
class SelectionRun:
    name: str
    spec_json: str
    ablation: str | None = None

    @property
    def spec(self) -> dict:
        return json.loads(self.spec_json)

    @property
    def is_oracle(self) -> bool:
        return self.spec.get("kind") in ORACLE_KINDS

    @property
    def run_id(self) -> str:
        return f"{self.name}-{stable_hash(self.spec, 8)}"


@dataclass(frozen=True)
class GenerationRun:
    selection: SelectionRun
    prior: str
    model_label: str
    model_json: str
    timeline: bool
    task: str = "ad"  # ad | qa
    ablation: str | None = None
    generation_json: str = "{}"
    audio_context: str | None = None  # AD ablation: none | timeline | native

    @property
    def model(self) -> dict:
        return json.loads(self.model_json)

    @property
    def run_id(self) -> str:
        public = {k: v for k, v in self.model.items() if not k.endswith("_env") and k not in NON_RESULT_MODEL_KEYS}
        identity = [self.selection.run_id, self.prior, public, self.timeline, self.task,
                    json.loads(self.generation_json)]
        if self.audio_context is not None:
            identity.append(self.audio_context)
        h = stable_hash(identity, 8)
        return f"{self.task}-{self.selection.name}-{self.prior}-{self.model_label}-tl{int(self.timeline)}-{h}"


def _selection(spec: str | dict, ablation: str | None = None, name: str | None = None) -> SelectionRun:
    resolved = resolve_spec(spec)
    if name:
        resolved["name"] = name
    return SelectionRun(resolved["name"], json.dumps(resolved, sort_keys=True), ablation)


def core_selections(cfg: dict) -> list[SelectionRun]:
    a_slot = get(cfg, "matrix.a_selector", "A")
    names = [a_slot if s == "A" else s for s in get(cfg, "matrix.selectors")]
    overrides = get(cfg, "selectors", {}) or {}
    out = []
    for name in names:
        spec = overrides.get(name, name)
        if isinstance(spec, dict):
            spec = {"name": name, **spec}
        out.append(_selection(spec))
    return out


def model_specs(cfg: dict) -> list[tuple[str, dict]]:
    models = get(cfg, "models", {}) or {}
    labels = get(cfg, "matrix.models") or []
    missing = [m for m in labels if m not in models]
    if missing:
        raise ConfigError(f"matrix.models names undefined models: {missing}")
    return [(label, models[label]) for label in labels]


def core_generation_runs(cfg: dict, task: str = "ad") -> list[GenerationRun]:
    """selectors x priors x models (20 with the default 5 x 2 x 2)."""
    gen = get(cfg, "generation", {})
    runs = []
    for sel in core_selections(cfg):
        for prior in get(cfg, "matrix.priors"):
            for label, spec in model_specs(cfg):
                runs.append(GenerationRun(sel, prior, label, json.dumps(spec, sort_keys=True),
                                          bool(gen.get("timeline", False)), task,
                                          generation_json=json.dumps(_gen_identity(gen), sort_keys=True)))
    return runs


def _gen_identity(gen: dict) -> dict:
    identity = {k: gen.get(k) for k in ("max_new_tokens", "temperature", "word_limit_policy", "timeline_window_seconds")}
    if "native_audio_window_seconds" in gen:
        identity["native_audio_window_seconds"] = gen["native_audio_window_seconds"]
    return identity


# --- ablations (proposal §5.3) -------------------------------------------------

SUPERCLASS_SET = ("music", "impulsive", "vocal_nonspeech", "ambient", "other")


def controlled_ablation_spec(spec: str | dict, cfg: dict) -> dict:
    """Start from the frozen core spec, then change only the ablated factors."""
    import copy
    raw = {"base": spec} if isinstance(spec, str) else copy.deepcopy(spec)
    base_name = raw.get("base")
    custom = get(cfg, f"selectors.{base_name}")
    if not custom:
        return resolve_spec(spec)
    base = resolve_spec({"name": base_name, **custom} if isinstance(custom, dict) else custom)
    def merge(a, b):
        result = copy.deepcopy(a)
        for key, value in b.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result
    return resolve_spec(merge(base, {k: v for k, v in raw.items() if k != "base"}))


def ablation_selections(name: str, cfg: dict) -> list[SelectionRun]:
    """Selector-level variants for one ablation (1-4, 6-8)."""
    a = f"ablation{name}"
    def select(spec, ablation):
        return _selection(controlled_ablation_spec(spec, cfg), ablation)
    if name == "1":  # speech cues only vs non-speech events only vs both (audio-only selector)
        return [
            select({"base": "A_cue", "name": "A_cue-speech", "options": {"weights": {"nse": 0.0}, "candidate_streams": ["sc"]}}, a),
            select({"base": "A_cue", "name": "A_cue-nonspeech", "options": {"weights": {"sc": 0.0}, "candidate_streams": ["nse"]}}, a),
            select({"base": "A_cue", "name": "A_cue-both"}, a),
        ]
    if name == "2":  # trigger vs suppress, global and per class
        b = abs(float(get(cfg, "ablations.nse_magnitude", 0.5)))
        runs = [
            select({"base": "AV", "name": "AV-trigger", "options": {"weights": {"nse": b}}}, a),
            select({"base": "AV", "name": "AV-suppress", "options": {"weights": {"nse": -b}}}, a),
        ]
        for c in SUPERCLASS_SET:
            for sign, tag in ((1.0, "trig"), (-1.0, "supp")):
                comps = {k: 0.0 for k in ("music", "impulsive", "vocal_nonspeech", "ambient", "other",
                                          "clap_change", "loudness", "onset")}
                comps[c] = sign
                runs.append(select({"base": "AV", "name": f"AV-{c}-{tag}",
                                        "options": {"weights": {"nse": b}, "nse_components": comps}}, a))
        return runs
    if name == "3":  # Whisper-only audio events vs the tagger (needs both cue sets cached)
        return [
            select({"base": "A_cue", "name": "A_cue-tagger", "cue_set": "audio_cues.json"}, a),
            select({"base": "A_cue", "name": "A_cue-whispertags", "cue_set": "audio_cues.whisper_tags.json",
                    "options": {"nse_components": {"clap_change": 0.0}}}, a),
        ]
    if name == "4":  # event-time vs snap-to-gap placement
        return [
            select({"base": "AV", "name": "AV-snap", "options": {"placement": "snap"}}, a),
            select({"base": "AV", "name": "AV-event", "options": {"placement": "event"}}, a),
        ]
    if name == "6":  # per-class breakdown: one superclass at a time
        runs = []
        for c in ("music", "impulsive", "ambient"):
            comps = {k: 0.0 for k in ("music", "impulsive", "vocal_nonspeech", "ambient", "other",
                                      "clap_change", "loudness", "onset")}
            comps[c] = 1.0
            runs.append(select({"base": "A_cue", "name": f"A_cue-{c}-only",
                                    "options": {"weights": {"sc": 0.0}, "candidate_streams": ["nse"], "nse_components": comps}}, a))
        return runs
    if name == "7":  # flat vs hierarchical budget (run on the long-video subset)
        return [
            select({"base": "AV", "name": "AV-flat", "budget": "flat"}, a),
            select({"base": "AV", "name": "AV-hier", "budget": "hierarchical",
                        "hierarchy": get(cfg, "ablations.hierarchy", {}) or {}}, a),
        ]
    if name == "8":  # running memory vs local-window novelty
        return [
            select({"base": "AV", "name": "AV-memory", "options": {"novelty_mode": "memory"}}, a),
            select({"base": "AV", "name": "AV-memory-pure", "options": {"novelty_mode": "memory", "memory_blend": 1.0}}, a),
            select({"base": "AV", "name": "AV-local", "options": {"novelty_mode": "local"}}, a),
            select("V", a),
            select("V_mem", a),
        ]
    if name == "5":
        return []
    raise ConfigError(f"unknown ablation {name!r}; expected 1-8")


def ablation5_runs(cfg: dict, task: str = "ad") -> list[GenerationRun]:
    """Audio timeline absent vs present vs an audio-native model, on one model with
    the top selector (both named in config, chosen from dev results)."""
    selector = get(cfg, "ablations.timeline.selector")
    model_label = get(cfg, "ablations.timeline.model")
    native_label = get(cfg, "ablations.timeline.audio_native_model")
    prior = get(cfg, "ablations.timeline.prior", "P0")
    if not selector or not model_label:
        raise ConfigError("ablation 5 needs ablations.timeline.selector and ablations.timeline.model")
    models = get(cfg, "models", {}) or {}
    if model_label not in models or (native_label and native_label not in models):
        raise ConfigError("ablations.timeline names a model that is not defined under [models]")
    custom = get(cfg, f"selectors.{selector}")
    sel = _selection({"name": selector, **custom} if isinstance(custom, dict) else selector)
    gen = _gen_identity(get(cfg, "generation", {}))
    conditions = [(model_label, False, "none"), (model_label, True, "timeline")]
    if native_label:
        conditions.append((native_label, False, "native"))
    return [GenerationRun(sel, prior, label, json.dumps(models[label], sort_keys=True), timeline, task,
                          "ablation5", json.dumps(gen), audio_context=context if task == "ad" else None)
            for label, timeline, context in conditions]


def all_selection_runs(cfg: dict) -> list[SelectionRun]:
    seen: dict[str, SelectionRun] = {}
    for run in core_selections(cfg):
        seen.setdefault(run.run_id, run)
    for name in get(cfg, "ablations.enabled", []) or []:
        for run in ablation_selections(str(name), cfg):
            seen.setdefault(run.run_id, run)
    return list(seen.values())
