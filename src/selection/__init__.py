"""Cue-driven moment selection (proposal §3, §5.1, §5.5).

- cues.py: every cached cue stream resampled onto one time grid (CueBundle).
- two_stage.py: candidate generation -> gap snapping -> diverse top-K (AV, A_cue, V_mem).
- hierarchical.py: long-video chunking and event-density budget allocation.
- registry.py: one interface over U, V, A, AV, O and the ablation variants.

Salience (what is worth describing) and feasibility (where a description can
go) are computed separately and only combined in the final score.
"""
