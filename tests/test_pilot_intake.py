"""A native-video staging inventory must not imply verified capture acceptance."""
import copy
import json
from pathlib import Path

import pytest

from scripts.inspect_pilot_intake import inspect

REGISTRY = Path(__file__).resolve().parents[1]/'data/metadata/a3_pilot_01_intake.json'


def test_missing_media_remains_pending_and_fens_are_not_photo_verification(tmp_path):
    report = inspect(json.loads(REGISTRY.read_text()), tmp_path)
    assert report['missing_or_ambiguous_assets'] == 10
    assert report['status'] == 'awaiting_or_ambiguous_media'
    assert not report['protocol_accepted'] and not report['media_validation_performed']
    assert all(r['fen_syntax_valid'] for r in report['records'])
    assert [r['pieces'] for r in report['records']] == [32, 32, 30, 11, 7]


def test_filename_presence_does_not_validate_fake_video_bytes(tmp_path):
    registry = json.loads(REGISTRY.read_text())
    for row in registry['records']:
        for kind, name in [('video', 'original.MOV'), ('reference', 'reference.HEIC')]:
            folder = tmp_path/row[kind+'_directory'];folder.mkdir(parents=True)
            (folder/name).write_bytes(b'not actual media')
    report = inspect(registry, tmp_path)
    assert report['status'] == 'ready_for_media_validation'
    assert not report['protocol_accepted'] and not report['media_validation_performed']


def test_pilot_cannot_be_promoted_to_final_or_training_by_intake(tmp_path):
    original = json.loads(REGISTRY.read_text())
    for key, value in [('final_holdout', True), ('training_eligible', True)]:
        registry = copy.deepcopy(original);registry[key] = value
        with pytest.raises(ValueError, match='Development-only'):inspect(registry, tmp_path)


def test_duplicate_ids_and_media_path_escape_are_rejected(tmp_path):
    original = json.loads(REGISTRY.read_text())
    duplicate = copy.deepcopy(original);duplicate['records'][1]['capture_session_id'] = duplicate['records'][0]['capture_session_id']
    with pytest.raises(ValueError, match='Duplicate'):inspect(duplicate, tmp_path)
    original['records'][0]['video_directory'] = '../outside'
    with pytest.raises(ValueError, match='outside'):inspect(original, tmp_path)
