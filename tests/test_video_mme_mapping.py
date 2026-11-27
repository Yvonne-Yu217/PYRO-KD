import pytest
from scripts.map_video_mme_subset import map_subset


def fixture():
    return ({'dataset':'repo','revision':'pin','videos':[{'video_id':'001','videoID':'source'}]},
            {'repo':'repo','revision':'pin','archives':[{'archive':'chunk.zip','archive_bytes':123,
              'members':[{'name':'data/source.mp4','size':100}]}]})


def test_exact_id_mapping():
    result=map_subset(*fixture())
    assert result['complete'] and result['videos'][0]['member']['archive']=='chunk.zip'


def test_missing_is_retained_only_when_explicit():
    subset,inventory=fixture();inventory['archives']=[]
    with pytest.raises(ValueError,match='not located'): map_subset(subset,inventory)
    result=map_subset(subset,inventory,True)
    assert not result['complete'] and len(result['videos'])==1 and result['videos'][0]['member'] is None


def test_duplicate_source_and_changed_revision_fail():
    subset,inventory=fixture();inventory['archives']*=2
    with pytest.raises(ValueError,match='ambiguous'): map_subset(subset,inventory)
    inventory['revision']='other'
    with pytest.raises(ValueError,match='identities'): map_subset(subset,inventory)
