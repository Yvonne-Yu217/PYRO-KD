import random
import pytest
from src.utils.hierarchical_budget import segment_video,event_densities,allocate_budget


def test_boundary_segmentation_has_no_tiny_artificial_tail():
    assert segment_video(65,[],min_seconds=20,max_seconds=60)==[(0,32.5),(32.5,65)]
    assert segment_video(100,[5,30,31,90],min_seconds=20,max_seconds=60)==[(0,30),(30,65),(65,100)]


def test_boundary_events_count_exactly_once():
    assert event_densities([(0,10),(10,20)],[0,10,20])==[.1,.2]


def test_quiet_floor_and_capped_redistribution():
    assert allocate_budget([0,1,3],10,floor=1).quotas==[1,3,6]
    result=allocate_budget([100,1,0],10,floor=1,capacities=[2,10,10])
    assert result.quotas==[2,7,1]
    assert result.capacity_shortfall==0


def test_infeasibility_is_reported_without_redefining_k():
    result=allocate_budget([1,1],10,floor=1,capacities=[0,3])
    assert result.requested==10 and result.quotas==[0,3]
    assert result.capacity_shortfall==7 and result.floor_shortfall_segments==[0]
    with pytest.raises(ValueError,match='floor'):
        allocate_budget([1,1,1],2,floor=1)


def test_zero_density_ties_are_deterministic():
    assert allocate_budget([0,0,0],8,floor=1).quotas==[3,3,2]


def test_randomized_budget_conservation_and_capacity():
    rng=random.Random(0)
    for _ in range(200):
        n=rng.randint(1,20);k=rng.randint(n,100);caps=[rng.randint(0,20) for _ in range(n)]
        result=allocate_budget([rng.random() for _ in range(n)],k,floor=1,capacities=caps)
        assert sum(result.quotas)+result.capacity_shortfall==k
        assert sum(result.quotas)==min(k,sum(caps))
        assert all(min(1,c)<=q<=c for q,c in zip(result.quotas,caps))


def test_memory_and_separation_remain_global_under_segment_quotas():
    from src.utils.placed_selection import Candidate,select_placed
    candidates=[Candidate(0,0,2),Candidate(4,0,1),Candidate(6,0,.1)]
    result=select_placed(candidates,[(0,10)],2,required_words=1,max_advance=0,max_lag=0,
                        novelty_weight=1,embeddings=[[0,1],[0,1],[1,0]],
                        segment_ids=[0,1,1],segment_quotas=[1,1])
    assert [s.candidate_index for s in result.selected]==[0,2]
    assert result.budget_unselected_indices==[1]
