"""Reuse existing pilot attestations and corrected labels for D.3b FUNCTIONAL tests only."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.select_video_keyframes import read,write
from src.scanning.native_pilot import sha256_file


def prepare(output):
    catalog=read('data/metadata/a3_pilot_01_sources_v2.json');native=read(catalog['primary_manifest'])
    labels=read('docs/evaluation/fusion_d3_summary_v1.json')['label_version']
    attest=read(catalog['owner_attestations']);rows=[]
    for s,n,a in zip(catalog['sessions'],native['sessions'],attest['sessions']):
        sid=s['capture_session_id']
        if sid!=n['capture_session_id'] or sid!=a['capture_session_id'] or a['photo_fen_agreement']!='owner_confirmed' or a['unchanged_position_full_recording']!='owner_confirmed':raise ValueError('Missing existing owner confirmation')
        rows.append(dict(session_id=sid,position_id=s['position_id'],position_group_id=s['position_group_id'],independence_group_id=s['independence_group_id'],
            capture_date='2026-10-19',source_video=s['primary_source'],reference=s['reference'],native_index=dict(path=n['index_path'],sha256=sha256_file(n['index_path'])),
            fen=labels['evaluated_fens'][sid],label_version=labels['correction_path']+'@'+labels['correction_sha256'],novel_position_attested=False,
            attestation=dict(reviewer=attest['actor'],reviewer_kind='human',photo_fen_agreement=True,unchanged_position=True,
                original_attestation_path=catalog['owner_attestations'],original_attestation_sha256=catalog['owner_attestations_sha256'],label_supplement=labels['correction_path']),
            board_id=n.get('board_id') or 'unknown',lighting=n.get('lighting') or 'unknown',environment=n.get('environment') or 'unknown',
            device_metadata=s['primary_source']['container_metadata'],viewpoint_coverage=dict(sides='see existing D.1 metadata',elevation='unknown')))
    write(output,dict(schema_version='fusion_independent_registry_d3b_v1',final_holdout=False,training_eligible=False,outcomes_inspected=True,used_for_tuning=True,
        functional_development_only=True,sessions=rows))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();prepare(a.output)
