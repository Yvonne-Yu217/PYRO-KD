"""One optional model repair, preserving both drafts and never truncating text."""
import json
import time


def repair_once(initial, prompt, word_limit, generate):
    attempts = [{'prompt': prompt, **initial}]
    result = dict(initial)
    elapsed = 0.
    if initial.get('word_limit_exceeded') and not initial.get('hit_token_limit'):
        repair_prompt = (
            prompt + '\nA previous draft exceeded the word limit: '
            + json.dumps(initial['output'], ensure_ascii=False)
            + f'\nRewrite it as a complete description or phrase of at most {word_limit} '
              'space-separated words. Remove optional details. Return only the shortened description.'
        )
        start = time.perf_counter()
        try:
            repaired = generate(repair_prompt)
            attempts.append({'prompt': repair_prompt, **repaired})
            result = dict(repaired)
        except Exception as exc:
            attempts.append({'prompt': repair_prompt, 'error': f'repair: {exc!r}'})
            result['repair_error'] = attempts[-1]['error']
        elapsed = time.perf_counter() - start
    result.update(generation_attempts=attempts, generation_calls=len(attempts),
                  total_new_tokens=sum(a.get('new_tokens', 0) for a in attempts),
                  repair_latency_s=elapsed)
    return result
