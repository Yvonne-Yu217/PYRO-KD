"""Hand-set AudioSet superclasses for per-class trigger/suppress weights (RQ2).

The proposal asks for weights per superclass, not per label (Risk 6: few
parameters). Labels are matched by their AudioSet display name, exactly as the
tagger's own id2label reports it; names not listed fall back to keyword rules,
then to "other". Speech and silence are excluded from NSE (§3.1: "drop speech
and silence classes"), because VAD already covers them.

This mapping is a design choice, not data: override it in the experiment
config (audio_cues.superclass_overrides) and report the version used.
"""

from __future__ import annotations

TAXONOMY_VERSION = "hand-set-v1"

EXCLUDED = ("speech", "silence")
SUPERCLASSES = ("music", "impulsive", "vocal_nonspeech", "ambient", "other")

_EXACT: dict[str, str] = {
    # Speech (excluded from NSE)
    "Speech": "speech",
    "Male speech, man speaking": "speech",
    "Female speech, woman speaking": "speech",
    "Child speech, kid speaking": "speech",
    "Conversation": "speech",
    "Narration, monologue": "speech",
    "Babbling": "speech",
    "Speech synthesizer": "speech",
    "Whispering": "speech",
    "Hubbub, speech noise, speech babble": "speech",
    # Silence (excluded)
    "Silence": "silence",
    # Music
    "Music": "music",
    "Musical instrument": "music",
    "Singing": "music",
    "Song": "music",
    "Background music": "music",
    "Theme music": "music",
    "Soundtrack music": "music",
    "Jingle (music)": "music",
    # Impulsive events
    "Applause": "impulsive",
    "Clapping": "impulsive",
    "Vehicle horn, car horn, honking": "impulsive",
    "Car alarm": "impulsive",
    "Door": "impulsive",
    "Slam": "impulsive",
    "Knock": "impulsive",
    "Bang": "impulsive",
    "Explosion": "impulsive",
    "Gunshot, gunfire": "impulsive",
    "Glass": "impulsive",
    "Shatter": "impulsive",
    "Smash, crash": "impulsive",
    "Thump, thud": "impulsive",
    "Bark": "impulsive",
    "Doorbell": "impulsive",
    "Alarm": "impulsive",
    "Beep, bleep": "impulsive",
    "Whistle": "impulsive",
    "Splash, splatter": "impulsive",
    # Vocal, non-speech
    "Laughter": "vocal_nonspeech",
    "Baby laughter": "vocal_nonspeech",
    "Giggle": "vocal_nonspeech",
    "Chuckle, chortle": "vocal_nonspeech",
    "Cheering": "vocal_nonspeech",
    "Crying, sobbing": "vocal_nonspeech",
    "Screaming": "vocal_nonspeech",
    "Gasp": "vocal_nonspeech",
    "Sigh": "vocal_nonspeech",
    "Crowd": "vocal_nonspeech",
    # Ambient
    "Wind": "ambient",
    "Rain": "ambient",
    "Water": "ambient",
    "Stream": "ambient",
    "Ocean": "ambient",
    "Traffic noise, roadway noise": "ambient",
    "Inside, small room": "ambient",
    "Inside, large room or hall": "ambient",
    "Outside, urban or manmade": "ambient",
    "Outside, rural or natural": "ambient",
    "Environmental noise": "ambient",
    "Noise": "ambient",
    "White noise": "ambient",
    "Static": "ambient",
    "Hum": "ambient",
    "Bird": "ambient",
    "Vehicle": "ambient",
    "Car": "ambient",
}

# Applied in order when a label is not listed above (lower-cased substring).
_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("speech", "speech"),
    ("speaking", "speech"),
    ("music", "music"),
    ("guitar", "music"),
    ("piano", "music"),
    ("drum", "music"),
    ("violin", "music"),
    ("synthesizer", "music"),
    ("orchestra", "music"),
    ("applause", "impulsive"),
    ("clap", "impulsive"),
    ("knock", "impulsive"),
    ("bang", "impulsive"),
    ("slam", "impulsive"),
    ("crash", "impulsive"),
    ("gun", "impulsive"),
    ("explosion", "impulsive"),
    ("horn", "impulsive"),
    ("laugh", "vocal_nonspeech"),
    ("cheer", "vocal_nonspeech"),
    ("scream", "vocal_nonspeech"),
    ("cry", "vocal_nonspeech"),
    ("noise", "ambient"),
    ("wind", "ambient"),
    ("rain", "ambient"),
    ("water", "ambient"),
    ("inside,", "ambient"),
    ("outside,", "ambient"),
)


def superclass_of(label: str, overrides: dict[str, str] | None = None) -> str:
    """The superclass of one AudioSet display name (see module docstring)."""
    if overrides and label in overrides:
        return overrides[label]
    if label in _EXACT:
        return _EXACT[label]
    lowered = label.lower()
    for keyword, superclass in _KEYWORDS:
        if keyword in lowered:
            return superclass
    return "other"


def is_nse_label(label: str, overrides: dict[str, str] | None = None) -> bool:
    """False for speech and silence labels, which NSE drops."""
    return superclass_of(label, overrides) not in EXCLUDED
