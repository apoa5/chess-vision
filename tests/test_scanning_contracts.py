import pytest
from src.scanning.contracts import BoardProfile, FrameObservation, ScanSnapshot


def frame(**changes):
    values=dict(session_id='scan1',frame_id='frame1',timestamp_ms=12.5,board_epoch=0,orientation_version=1,geometry_version=1,width=1920,height=1080,h_board_to_frame=(1.,0.,0.,0.,-1.,32.,0.,0.,1.),source_sha256='a'*64,orientation_source='manual')
    return FrameObservation(**(values|changes))


def test_physical_profile_preserves_border():
    board=BoardProfile()
    assert board.playing_area_cm==32
    assert board.total_cm==39


def test_snapshot_has_64_unobserved_squares():
    state=ScanSnapshot('scan1').to_dict()
    assert len(state['squares'])==64
    assert set(state['squares'].values())=={'unobserved'}
    assert state['tracking']=='uninitialized'


@pytest.mark.parametrize('changes',[{'width':0},{'timestamp_ms':float('nan')},{'board_epoch':-1},{'source_sha256':'bad'},{'orientation_source':'guessed'},{'h_board_to_frame':(1.,)},{'schema_version':'future'}])
def test_rejects_invalid_frame_provenance(changes):
    with pytest.raises(ValueError):frame(**changes)


def test_frame_serializes_source_dimensions_and_versions():
    result=frame().to_dict()
    assert result['orientation_source']=='manual'
    assert result['width']==1920
    assert result['geometry_version']==1


def test_rejects_missing_square_and_unknown_resolution():
    with pytest.raises(ValueError):ScanSnapshot('scan1',squares={'a1':'resolved'})
    squares=ScanSnapshot('scan1').squares.copy()
    squares['a1']='perfect'
    with pytest.raises(ValueError):ScanSnapshot('scan1',squares=squares)
