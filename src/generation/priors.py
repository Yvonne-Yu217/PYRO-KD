"""P1 prior: AD guideline summary plus 3 professional exemplars (proposal §5.2, RQ3).

The guideline summary below is our own condensed paraphrase of common AD
practice (describe what is seen, present tense, objective, no restating
dialogue, prioritize what matters for following the story, fit the pause). It
is prompt text, versioned, and shared by every P1 run.

Exemplars are NOT written here. They must be real professional AD lines,
drawn from the exemplar pool (MTG-AD validation videos outside dev, never
test) by scripts/build_p1_exemplars.py into configs/p1_exemplars.json, which
records each line's source video and time. If that file is missing, P1 cannot
run and says so (MissingAsset), instead of falling back to invented examples.
"""

from __future__ import annotations

import json
from pathlib import Path

GUIDELINES_VERSION = "p1-guidelines-v1"
GUIDELINES = (
    "Audio description guidelines:\n"
    "- Describe what is visible and relevant to following the video: who, what, where, and key actions.\n"
    "- Use present tense and plain, specific words; name on-screen text when it matters.\n"
    "- Be objective: describe, do not interpret emotions, intentions, or quality.\n"
    "- Do not repeat what the dialogue or sounds already make clear.\n"
    "- Fit the pause: say the most important thing first and stay within the word limit."
)
DEFAULT_EXEMPLARS_PATH = Path("configs/p1_exemplars.json")
NUM_EXEMPLARS = 3


class MissingAsset(RuntimeError):
    """A required non-code asset (exemplars, hand labels, credentials) is not available."""


def load_exemplars(path: str | Path = DEFAULT_EXEMPLARS_PATH) -> dict:
    """{"version", "source", "exemplars": [{"text", "video_id", "time", ...}]} with exactly 3 exemplars."""
    path = Path(path)
    if not path.is_file():
        raise MissingAsset(
            f"P1 exemplars not found at {path}. Build them from the MTG-AD exemplar pool with "
            "`python scripts/build_p1_exemplars.py` (needs data/mtg-ad annotations). "
            "Exemplars are never invented."
        )
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MissingAsset(f"P1 exemplars at {path} are unreadable: {exc}") from exc
    exemplars = record.get("exemplars") if isinstance(record, dict) else None
    if (
        not isinstance(exemplars, list) or len(exemplars) != NUM_EXEMPLARS
        or not all(isinstance(e, dict) and isinstance(e.get("text"), str) and e["text"].strip()
                   and isinstance(e.get("video_id"), str) for e in exemplars)
    ):
        raise MissingAsset(f"{path} must hold exactly {NUM_EXEMPLARS} exemplars with text and source video_id")
    return record


def prior_block(exemplars: dict) -> str:
    lines = "\n".join(f'{i}. "{e["text"].strip()}"' for i, e in enumerate(exemplars["exemplars"], 1))
    return f"{GUIDELINES}\n\nExamples of professional audio descriptions:\n{lines}\n\n"


def select_exemplars(candidates: list[dict], forbidden_video_ids: set[str], seed: int = 0, max_words: int = 15) -> list[dict]:
    """Deterministic pick of 3 exemplar lines from real annotations.

    candidates: [{"video_id", "time", "text", "category"}] from the exemplar pool.
    Lines from forbidden videos (dev and test) are excluded; lines must be 4..max_words
    words so they model concision. One line per category where possible, seeded.
    """
    import random

    pool = [
        c for c in candidates
        if c["video_id"] not in forbidden_video_ids and 4 <= len(c["text"].split()) <= max_words
    ]
    if len(pool) < NUM_EXEMPLARS:
        raise MissingAsset("fewer than 3 eligible exemplar lines in the pool")
    rng = random.Random(seed)
    pool.sort(key=lambda c: (c.get("category", ""), c["video_id"], c["time"]))
    rng.shuffle(pool)
    chosen, categories, videos = [], set(), set()
    for c in pool:
        if c.get("category") in categories or c["video_id"] in videos:
            continue
        chosen.append(c)
        categories.add(c.get("category"))
        videos.add(c["video_id"])
        if len(chosen) == NUM_EXEMPLARS:
            break
    for c in pool:  # fewer categories than 3: fill from distinct videos
        if len(chosen) == NUM_EXEMPLARS:
            break
        if c["video_id"] not in videos:
            chosen.append(c)
            videos.add(c["video_id"])
    if len(chosen) < NUM_EXEMPLARS:
        raise MissingAsset("could not find 3 exemplars from distinct videos")
    return chosen
