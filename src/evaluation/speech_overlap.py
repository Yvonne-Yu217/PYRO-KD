"""Collision metrics for measured rendered AD intervals (seconds, half-open).

Do not pass word-count duration estimates as rendered durations. VAD and ASR
are reported separately and as their union; overlapping detector intervals are
merged so a collision is never counted twice within a metric.
"""
import math


def union(intervals):
    checked = []
    for start, end in intervals:
        if not all(math.isfinite(x) for x in (start, end)) or not 0 <= start <= end:
            raise ValueError('intervals must be finite and nonnegative with start <= end')
        if end > start:
            checked.append((start, end))
    merged = []
    for start, end in sorted(checked):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def intersection_seconds(left, right):
    left, right = union(left), union(right)
    i = j = 0
    total = 0.
    while i < len(left) and j < len(right):
        a, b = left[i], right[j]
        total += max(0., min(a[1], b[1]) - max(a[0], b[0]))
        if a[1] <= b[1]:
            i += 1
        else:
            j += 1
    return total


def evaluate_intervals(narrations, vad, words, video_duration):
    """Keep per-description collision and unique timeline collision distinct.

    narrations contain id/start/duration measured after tempo conversion.
    Out-of-video duration is reported, never silently truncated or counted as
    successfully used silence. Detector intervals must lie within the video.
    """
    if not math.isfinite(video_duration) or video_duration <= 0:
        raise ValueError('positive finite video duration required')
    tracks = {'vad': union(vad), 'whisper': union(words)}
    if any(end > video_duration for track in tracks.values() for _, end in track):
        raise ValueError('speech interval exceeds video duration')
    tracks['union'] = union(tracks['vad'] + tracks['whisper'])
    rows, ids, intervals = [], set(), []
    for item in narrations:
        ident, start, duration = item['id'], item['start'], item['duration']
        if not isinstance(ident, str) or not ident or ident in ids:
            raise ValueError('unique nonempty narration IDs required')
        if not all(math.isfinite(x) for x in (start, duration)) or start < 0 or duration <= 0:
            raise ValueError('finite nonnegative start and positive rendered duration required')
        ids.add(ident)
        end = start + duration
        if not math.isfinite(end):
            raise ValueError('nonfinite narration end')
        intervals.append((start, end))
        collisions = {name: intersection_seconds([(start, end)], track) for name, track in tracks.items()}
        outside = max(0., end - max(start, video_duration))
        rows.append({'id': ident, 'start': start, 'end': end, 'duration': duration,
                     'speech_overlap_seconds': collisions,
                     'out_of_video_seconds': outside,
                     'speech_free_in_video_seconds': max(0., duration-outside-collisions['union']),
                     'fits_speech_free_video': collisions['union'] == 0 and outside == 0})
    merged = union(intervals)
    total = sum(r['duration'] for r in rows)
    unique = sum(end-start for start, end in merged)
    return {'rows': rows, 'count': len(rows), 'rendered_seconds_sum': total,
            'narration_union_seconds': unique,
            'narration_excess_layer_seconds': max(0., total-unique),
            'speech_overlap_seconds_sum': {name: sum(r['speech_overlap_seconds'][name] for r in rows) for name in tracks},
            'speech_overlap_seconds_timeline': {name: intersection_seconds(merged, track) for name, track in tracks.items()},
            'out_of_video_seconds_sum': sum(r['out_of_video_seconds'] for r in rows),
            'fraction_descriptions_fitting': sum(r['fits_speech_free_video'] for r in rows)/len(rows) if rows else None}


def accelerate_rendered_audio(source, destination):
    """Materialize 1.2x playback and measure PCM samples, retaining encoder tails.

    Input must be the unaccelerated TTS waveform. Never infer duration from text
    length, strip silence, or overwrite an existing rendered artifact.
    """
    import subprocess
    import wave
    from pathlib import Path
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-n', '-i', str(source),
                    '-map', '0:a:0', '-vn', '-af', 'atempo=1.2', '-ac', '1',
                    '-ar', '16000', '-c:a', 'pcm_s16le', str(destination)],
                   check=True, capture_output=True)
    with wave.open(str(destination), 'rb') as wav:
        duration = wav.getnframes()/wav.getframerate()
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('empty rendered audio')
    return duration
