import numpy as np
import pytest
from src.utils.placed_selection import Candidate, select_placed


def select(candidates,gaps,k,**kwargs):
    options=dict(required_words=3,max_advance=0,max_lag=0,novelty_weight=1)
    options.update(kwargs)
    return select_placed(candidates,gaps,k,**options)


def test_infeasible_event_does_not_contaminate_memory():
    candidates=[Candidate(0,0,100),Candidate(4,0,2),Candidate(6,0,1),Candidate(8,0,.1)]
    result=select(candidates,[(4,12)],2,embeddings=[[0,1],[1,0],[1,0],[0,1]])
    assert [s.candidate_index for s in result.selected]==[1,3]
    assert result.infeasible_indices==[0]
    assert result.budget_unselected_indices==[2]
    assert result.shortfall==0


def test_separation_uses_placement_not_event_time():
    result=select([Candidate(1,3),Candidate(3,2),Candidate(7,1)],[(4,10)],3,max_lag=3)
    assert [s.placement.time for s in result.selected]==[4,7]
    assert result.separation_rejected_indices==[1]
    assert result.shortfall==1
    assert all(s.placement.word_limit >= 3 for s in result.selected)


def test_signed_base_scores_and_stable_ties():
    result=select([Candidate(1,-1),Candidate(3,-2),Candidate(5,-1)],[(0,9)],2,novelty_weight=0)
    assert [s.candidate_index for s in result.selected]==[0,2]
    assert [s.rank for s in result.selected]==[0,1]


def test_chronological_output_retains_selection_rank():
    result=select([Candidate(1,1),Candidate(3,2)],[(0,9)],2)
    assert [s.rank for s in result.selected]==[1,0]


def test_missing_gaps_never_create_fake_placements():
    result=select([Candidate(1,1)],[],3)
    assert result.shortfall==3
    assert result.infeasible_indices==[0]


@pytest.mark.parametrize('options',[{'k':0},{'required_words':0},{'max_lag':-1},
    {'novelty_weight':float('nan')},{'min_separation':-1},{'embeddings':[[0,0]]}])
def test_invalid_protocol_fails(options):
    k=options.pop('k',1)
    with pytest.raises(ValueError):
        select([Candidate(1,1)],[(0,9)],k,**options)


def test_all_candidates_accounted_for():
    result=select([Candidate(i,i) for i in range(15)],[(4,12)],2)
    groups=[[s.candidate_index for s in result.selected],result.infeasible_indices,
            result.separation_rejected_indices,result.budget_unselected_indices]
    assert sorted(i for group in groups for i in group)==list(range(15))


def test_gap_reward_uses_remaining_capacity_and_changes_rank():
    candidates=[Candidate(1,1),Candidate(8,1.1)]
    plain=select(candidates,[(0,10)],1,novelty_weight=0)
    scored=select(candidates,[(0,10)],1,novelty_weight=0,gap_weight=1,gap_target_words=6)
    assert plain.selected[0].candidate_index == 1
    assert scored.selected[0].candidate_index == 0
    chosen=scored.selected[0]
    assert chosen.gap_feasibility == pytest.approx(27/(27+6))
    assert chosen.score == pytest.approx(1+chosen.gap_feasibility)
    assert chosen.placement.word_limit == 27  # scoring target does not cap generation


def test_gap_score_is_computed_after_snapping_and_zero_weight_is_ablation():
    candidates=[Candidate(1,2),Candidate(8,1)]
    result=select(candidates,[(4,10)],2,max_lag=3,gap_weight=1,gap_target_words=18)
    first=result.selected[0]
    assert first.placement.time == 4 and first.gap_feasibility == .5
    plain=select(candidates,[(4,10)],2,max_lag=3)
    disabled=select(candidates,[(4,10)],2,max_lag=3,gap_weight=0,gap_target_words=18)
    assert [(s.candidate_index,s.score) for s in plain.selected] == [(s.candidate_index,s.score) for s in disabled.selected]


@pytest.mark.parametrize('options',[{'gap_weight':-1},{'gap_weight':float('nan')},
    {'gap_weight':1},{'gap_target_words':0},{'gap_target_words':True}])
def test_invalid_gap_score_configuration_fails(options):
    with pytest.raises(ValueError):
        select([Candidate(1,1)],[(0,9)],1,**options)
