"""Explicit frozen memberships for cache preparation; never inspect labels."""


def unique_ids(values, label):
    if not isinstance(values, list) or not values:
        raise ValueError(f'{label}: nonempty frozen membership required')
    if any(not isinstance(v, str) or not v or v in {'.', '..'} or '/' in v or '\\' in v for v in values):
        raise ValueError(f'{label}: invalid video ID')
    if len(set(values)) != len(values):
        raise ValueError(f'{label}: duplicate IDs')
    return values


def frozen_video_ids(config, subset='dev'):
    """REFRAMED dev/test or Video-MME long; caller explicitly chooses schema."""
    if subset in {'dev', 'test'}:
        if config.get('dataset') != 'igorsterner/reframed':
            raise ValueError('dev/test requires the frozen REFRAMED split config')
        dev = unique_ids([r['video_id'] for r in config['clips']], 'dev')
        test = unique_ids(config['test']['video_ids'], 'test')
        if set(dev) & set(test):
            raise ValueError('dev/test memberships overlap')
        return dev if subset == 'dev' else test
    if subset == 'long':
        if config.get('dataset') != 'lmms-eval/Video-MME' or config.get('official_split') != 'test':
            raise ValueError('long requires the frozen Video-MME subset config')
        return unique_ids([r['video_id'] for r in config['videos']], 'long')
    raise ValueError('subset must be dev, test or long')


def egoschema_ids(config, split='pilot'):
    pilot = unique_ids(config['pilot_q_uids'], 'pilot')
    heldout = unique_ids(config['heldout_q_uids'], 'heldout')
    if len(pilot) != 10 or len(heldout) != config['num_heldout']:
        raise ValueError('frozen EgoSchema membership counts differ')
    if set(pilot) & set(heldout):
        raise ValueError('pilot/heldout memberships overlap')
    if split not in {'pilot', 'heldout'}:
        raise ValueError('split must be pilot or heldout')
    return pilot if split == 'pilot' else heldout
