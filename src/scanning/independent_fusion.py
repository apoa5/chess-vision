"""Registration guards for offline evaluation cohorts, never final-holdout access."""
from datetime import date
import re
from src.chess.fen import board_from_fen, placement_fen


def validate_registry(registry, known, *, functional_development=False):
    if registry.get('schema_version') != 'fusion_independent_registry_d3b_v1' or registry.get('final_holdout') is not False or registry.get('training_eligible') is not False:
        raise ValueError('Evaluation-only registry required; final holdout prohibited')
    if not functional_development and (registry.get('outcomes_inspected') is not False or registry.get('used_for_tuning') is not False):
        raise ValueError('Inspected or tuned sessions cannot be independent validation')
    sessions=registry.get('sessions',[])
    if not sessions: raise ValueError('No collected sessions; independent validation pending')
    ids=set();sources=set();positions={};placements={};position_labels={}
    for s in sessions:
        sid=s['session_id']; placement=placement_fen(board_from_fen(s['fen']))
        if len(s['fen'].split())!=6: raise ValueError('Complete exact FEN required')
        if sid in ids or s['source_video']['sha256'] in sources: raise ValueError('Duplicate session/recording representation')
        ids.add(sid); sources.add(s['source_video']['sha256'])
        if not functional_development and (sid in known['sessions'] or s['source_video']['sha256'] in known['sources'] or placement in known['placements']):
            raise ValueError('Pilot/training placement or recording reused as independent data')
        if s.get('novel_position_attested') is not True and not functional_development: raise ValueError('New-position human attestation required')
        for identity in ('position_id','position_group_id','independence_group_id','capture_date','board_id','lighting','environment','label_version'):
            if not str(s.get(identity) or '').strip(): raise ValueError('Missing '+identity)
        date.fromisoformat(s['capture_date'])
        if position_labels.setdefault(s['position_id'],placement)!=placement: raise ValueError('Position ID has inconsistent FEN placement')
        if positions.setdefault(s['position_id'],s['position_group_id']) != s['position_group_id'] or placements.setdefault(placement,s['position_group_id']) != s['position_group_id']:
            raise ValueError('Same position/placement must stay grouped')
        a=s.get('attestation',{})
        if not a.get('reviewer') or a.get('reviewer_kind')!='human' or any(a.get(k) is not True for k in ('photo_fen_agreement','unchanged_position')):
            raise ValueError('Explicit human photo/FEN and stability confirmation required')
        for key in ('source_video','reference','native_index'):
            if not s[key].get('path') or not re.fullmatch(r'[0-9a-f]{64}',str(s[key].get('sha256',''))): raise ValueError('Missing source/reference/index provenance')
        if not s.get('device_metadata') or not s.get('viewpoint_coverage'):
            raise ValueError('Device and coarse viewpoint metadata required; use explicit unknown values')
    return dict(sessions=len(sessions),distinct_position_groups=len(set(positions.values())),
                capture_dates=len({s['capture_date'] for s in sessions}),
                lighting_conditions=sorted({s['lighting'] for s in sessions}),
                environment_ids=sorted({str(s['environment']) for s in sessions}),board_ids=sorted({s['board_id'] for s in sessions}),
                independence_groups=sorted({s['independence_group_id'] for s in sessions}))
