"""Explicit C.4 review gate. Completion and model predictions never certify labels."""
import hashlib,json
from src.training.multidataset import validate_board

CHECKS=('fen_agreement','unchanged_position','semantic_orientation','corners','canonical_mapping','crop_alignment','all_crops_reviewed')

def binding(frame):
    return hashlib.sha256(json.dumps({k:frame[k] for k in ('frame_id','source_sha256','pts_time','decoded_pixel_sha256','geometry_sha256','fen')},sort_keys=True).encode()).hexdigest()

def apply_reviews(packet,reviews):
    if reviews.get('packet_sha256')!=hashlib.sha256(json.dumps(packet,sort_keys=True).encode()).hexdigest():raise ValueError('Review packet provenance mismatch')
    if not reviews.get('reviewer','').strip():raise ValueError('Attributed human reviewer required')
    result=[]
    for frame in packet['frames']:
        b=json.loads(json.dumps(frame['board']));review=reviews.get('frames',{}).get(frame['frame_id'])
        if review:
            if review.get('binding')!=binding(frame):raise ValueError('Review belongs to different video/FEN/geometry')
            if review.get('status')=='verified':
                if not all(review.get('checks',{}).get(c) is True for c in CHECKS):raise ValueError('All geometry, label and crop checks required')
                flags=review.get('ambiguous_squares',[])
                if set(flags)-{o['square'] for o in b['observations']}:raise ValueError('Invalid square flag')
                b['geometry']['status']='human_verified';b['annotation'].update(status='human_verified',checks=review['checks'],reviewer=reviews['reviewer'])
                for o in b['observations']:
                    if o['square'] in flags:o['visibility']='ambiguous'
                if not frame['protected_overlap']:b['inclusion']=dict(status='included',reason='explicit_human_verification')
            elif review.get('status')=='rejected':b['inclusion']=dict(status='quarantine',reason='human_rejected_geometry_or_labels')
            elif review.get('status')=='pending':pass
            else:raise ValueError('Unknown review decision')
        result.append(validate_board(b))
    return result
