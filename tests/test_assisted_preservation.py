import pytest
from scripts.check_assisted_mvp1 import check_snapshot, sha256


def test_snapshot_checks_every_file_and_only_authorized_edits(tmp_path):
    files=[tmp_path/str(i) for i in range(4)]
    for p in files:p.write_bytes(b'original')
    baseline={str(p):sha256(p) for p in files}
    files[1].write_bytes(b'app edit');files[2].write_bytes(b'forbidden edit');files[3].unlink()
    report=check_snapshot(baseline,{str(files[1])})
    assert report['checked']==4 and report['unchanged']==1
    assert len(report['authorized_application_changes'])==1
    assert report['unexpected_changes'][0]['path']==str(files[2])
    assert report['missing']==[str(files[3])]
    assert not report['checkpoints_and_original_media_unchanged']


def test_holdout_paths_are_never_opened():
    with pytest.raises(ValueError,match='Holdout path access prohibited'):
        check_snapshot({'do-not-open-final-holdout': '0'*64},set())


def test_unchanged_inputs_complete_the_count(tmp_path):
    files=[tmp_path/str(i) for i in range(3)]
    for p in files:p.write_bytes(b'preserve')
    report=check_snapshot({str(p):sha256(p) for p in files},set())
    assert report['unchanged']==report['checked']==3
    assert report['checkpoints_and_original_media_unchanged']
