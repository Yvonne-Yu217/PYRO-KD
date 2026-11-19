"""Three-stream audio stack (proposal §2-§3): VAD-gated ASR, audio-event tagging
with CLAP embeddings, and the cue scores derived from them (NSE, speech cues).

Silero VAD itself stays in src/utils/audio_features.py (it defines the legal
gaps); everything here is additional salience, never feasibility.
"""
