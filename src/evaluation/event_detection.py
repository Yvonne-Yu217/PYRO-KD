"""Window-level tagger precision/recall on explicitly reviewed human intervals.

This evaluates class presence in the tagger's audio window, not onset accuracy.
Unreviewed audio is excluded, never silently counted as a negative label.
"""
import math
import numpy as np


def intervals(rows, duration):
    result = []
    for row in rows:
        if len(row) != 2:
            raise ValueError('interval must contain start and end')
        a, b = row
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                   and math.isfinite(x) for x in (a, b)) or not 0 <= a < b <= duration:
            raise ValueError('invalid annotation interval')
        result.append((a, b))
    merged = []
    for a, b in sorted(result):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return merged


def metrics(tp, fp, fn, tn):
    return dict(tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                precision=float(tp / (tp + fp)) if tp + fp else None,
                recall=float(tp / (tp + fn)) if tp + fn else None,
                f1=float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else None)


def score_event_windows(times, probabilities, labels, annotation, thresholds, window_seconds):
    """Use any positive-duration event/window intersection as class presence.

    Every scored window's real (unpadded) support must be fully reviewed for
    every class in thresholds. Unknown/ambiguous intervals exclude intersecting
    windows. Exact boundary contact alone is not a positive intersection.
    """
    duration = annotation['duration']
    if (not isinstance(duration, (int, float)) or isinstance(duration, bool)
            or not math.isfinite(duration) or duration <= 0
            or not math.isfinite(window_seconds) or window_seconds <= 0):
        raise ValueError('positive finite duration and window required')
    if (not thresholds or len(set(labels)) != len(labels)
            or not set(thresholds) <= set(labels)
            or set(annotation['reviewed_classes']) != set(thresholds)
            or len(annotation['reviewed_classes']) != len(thresholds)):
        raise ValueError('explicit unique reviewed class vocabulary required')
    if any(isinstance(x, bool) or not isinstance(x, (float, int))
           or not math.isfinite(x) or not 0 <= x <= 1 for x in thresholds.values()):
        raise ValueError('thresholds must be finite probabilities')
    t = np.asarray(times, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if (t.ndim != 1 or not len(t) or p.shape != (len(t), len(labels))
            or not np.isfinite(t).all() or not np.isfinite(p).all()
            or (t < 0).any() or (t >= duration).any() or (np.diff(t) <= 0).any()
            or (p < 0).any() or (p > 1).any()):
        raise ValueError('invalid tagger arrays')
    reviewed = intervals(annotation['reviewed_intervals'], duration)
    unknown = intervals(annotation.get('unknown_intervals', []), duration)
    events = {label: [] for label in thresholds}
    for event in annotation['events']:
        if event['label'] not in events:
            raise ValueError('event outside reviewed class vocabulary')
        a, b = intervals([[event['start'], event['end']]], duration)[0]
        if not any(x <= a and b <= y for x, y in reviewed):
            raise ValueError('event outside reviewed audio')
        events[event['label']].append((a, b))
    supports = [(max(0., x-window_seconds/2), min(duration, x+window_seconds/2)) for x in t]
    eligible = [i for i, (a, b) in enumerate(supports)
                if any(x <= a and b <= y for x, y in reviewed)
                and not any(a < y and x < b for x, y in unknown)]
    if not eligible:
        raise ValueError('no fully reviewed windows to score')
    per_class = {}
    for label, threshold in thresholds.items():
        truth = np.array([any(a < y and x < b for x, y in events[label])
                          for a, b in (supports[i] for i in eligible)], dtype=bool)
        pred = p[eligible, labels.index(label)] >= threshold
        per_class[label] = metrics(np.sum(pred & truth), np.sum(pred & ~truth),
                                   np.sum(~pred & truth), np.sum(~pred & ~truth))
    micro = metrics(*(sum(row[k] for row in per_class.values()) for k in ('tp', 'fp', 'fn', 'tn')))
    return dict(scored_windows=len(eligible), excluded_windows=len(t)-len(eligible),
                scored_indices=eligible, reviewed_seconds=sum(b-a for a, b in reviewed),
                per_class=per_class, micro=micro,
                scope='window-level class presence; not onset localization or human-label validation')
