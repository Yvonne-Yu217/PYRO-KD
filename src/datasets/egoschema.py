"""EgoSchema zero-shot QA annotations (downstream evaluation only).

Source: https://github.com/egoschema/EgoSchema. questions.json covers all
5,031 questions; subset_answers.json labels the public 500-question subset.
Only the subset can be scored locally; never use answers during selection
or generation, and never tune on items outside the declared pilot.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Iterable
import urllib.request

RAW_BASE_URL = "https://raw.githubusercontent.com/egoschema/EgoSchema/main"
ANNOTATION_FILES = ("questions.json", "subset_answers.json")
NUM_OPTIONS = 5


@dataclass(frozen=True)
class EgoSchemaItem:
    """One question; answer is None outside the public labeled subset."""

    q_uid: str
    question: str
    options: tuple[str, ...]
    google_drive_id: str | None
    answer: int | None

    @property
    def video_filename(self) -> str:
        """Released videos are named by question uid."""
        return f"{self.q_uid}.mp4"


def download_annotations(
    data_dir: str | Path,
    overwrite: bool = False,
) -> list[Path]:
    """Fetch questions.json and subset_answers.json into data_dir."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in ANNOTATION_FILES:
        target = data_dir / name
        if overwrite or not target.exists():
            with urllib.request.urlopen(f"{RAW_BASE_URL}/{name}") as response:
                target.write_bytes(response.read())
        paths.append(target)
    return paths


def parse_items(
    questions: Iterable[dict],
    answers: dict[str, int] | None = None,
) -> list[EgoSchemaItem]:
    """Build items from questions.json rows ('option 0'..'option 4')."""
    answers = answers or {}
    items = []
    for row in questions:
        options = tuple(row[f"option {i}"] for i in range(NUM_OPTIONS))
        answer = answers.get(row["q_uid"])
        if answer is not None and not 0 <= int(answer) < NUM_OPTIONS:
            raise ValueError(f"answer out of range for {row['q_uid']}: {answer}")
        items.append(EgoSchemaItem(
            q_uid=row["q_uid"],
            question=row["question"],
            options=options,
            google_drive_id=row.get("google_drive_id"),
            answer=None if answer is None else int(answer),
        ))
    return items


def load_items(data_dir: str | Path, labeled_only: bool = True) -> list[EgoSchemaItem]:
    """Load annotations; by default only the 500 public labeled items."""
    data_dir = Path(data_dir)
    questions = json.loads((data_dir / "questions.json").read_text())
    answers = json.loads((data_dir / "subset_answers.json").read_text())
    items = parse_items(questions, answers)
    if labeled_only:
        items = [item for item in items if item.answer is not None]
    return items


def split_pilot(
    items: Iterable[EgoSchemaItem],
    num_pilot: int = 10,
    seed: int = 0,
) -> tuple[list[EgoSchemaItem], list[EgoSchemaItem]]:
    """Deterministically split labeled items into (pilot, held-out eval).

    The proposal excludes pilot items from the remaining local evaluation.
    """
    ordered = sorted(items, key=lambda item: item.q_uid)
    pilot_ids = {item.q_uid for item in random.Random(seed).sample(ordered, num_pilot)}
    pilot = [item for item in ordered if item.q_uid in pilot_ids]
    rest = [item for item in ordered if item.q_uid not in pilot_ids]
    return pilot, rest
