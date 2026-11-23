"""Multiple-choice accuracy with frozen denominators and explicit failure coverage."""
from collections import Counter
from src.generation.qa import parse_answer
from src.generation.outputs import split_rows,is_failure


def score_qa(predictions, expected, num_options):
    """expected maps frozen question IDs to answer indices; generation never calls this.

    Retry errors may be superseded by a success. Duplicate successes, unknown
    IDs, mixed run fingerprints and disagreement between text/index are errors.
    Missing, invalid and failed predictions are incorrect in primary accuracy.
    """
    if num_options not in (4,5) or not expected:
        raise ValueError('nonempty expected questions with four/five options required')
    if any(not isinstance(ident,str) or not ident or isinstance(label,bool) or
           not isinstance(label,int) or not 0<=label<num_options for ident,label in expected.items()):
        raise ValueError('invalid evaluation labels')
    predictions=list(predictions)
    if any(row.get('id') not in expected for row in predictions):
        raise ValueError('prediction outside the declared evaluation split')
    fingerprints={row.get('run_fingerprint') for row in predictions}
    if len(fingerprints)>1: raise ValueError('mixed generation runs')
    if len({row.get('selector') for row in predictions})>1:
        raise ValueError('mixed selectors require separate accuracy reports')
    final,superseded=split_rows(predictions)
    if len({row['id'] for row in final})!=len(final): raise ValueError('duplicate successful predictions')
    by_id={row['id']:row for row in final}
    rows=[]
    for ident,label in expected.items():
        prediction=by_id.get(ident);choice=None
        if prediction is None: status='missing'
        elif is_failure(prediction):
            status=('invalid_answer' if str(prediction.get('error','')).startswith('invalid_answer:')
                    and not prediction.get('hit_token_limit') else 'generation_error')
        else:
            choice=parse_answer(prediction.get('output'),num_options)
            recorded=prediction.get('predicted_index')
            if recorded is not None and (isinstance(recorded,bool) or not isinstance(recorded,int) or recorded!=choice):
                raise ValueError('predicted index disagrees with strict text parse')
            status='valid' if choice is not None else 'invalid_answer'
        rows.append({'id':ident,'video_id':prediction.get('video_id',ident) if prediction else ident,
                     'answer_index':label,'predicted_index':choice,'status':status,
                     'correct':status=='valid' and choice==label})
    counts=Counter(row['status'] for row in rows)
    correct=sum(row['correct'] for row in rows)
    return {'complete':counts['valid']==len(expected),'expected':len(expected),
            'attempted':len(final),'valid_predictions':counts['valid'],'correct':correct,
            'accuracy':correct/len(expected),
            'accuracy_among_valid':correct/counts['valid'] if counts['valid'] else None,
            'status_counts':{status:counts[status] for status in ('valid','invalid_answer','generation_error','missing')},
            'superseded_error_attempts':len(superseded),
            'run_fingerprint':next(iter(fingerprints),None),'rows':rows,
            'denominator_policy':'all frozen expected questions; missing/invalid/errors count incorrect'}
