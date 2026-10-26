"""Lossless D.3b portable export; refuses to replace historical artifacts."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.select_video_keyframes import read,write
from src.scanning.native_pilot import sha256_file


def export(analysis,figures,functional,output):
    if output.exists(): raise ValueError('Fresh export directory required')
    raw=(analysis/'reliability_v1.json').read_bytes();r=json.loads(raw);output.mkdir(parents=True)
    with (output/'fusion_reliability_d3b_results_v1.json.gz').open('wb') as stream:
        with gzip.GzipFile(filename='',mode='wb',fileobj=stream,mtime=0) as zipped:zipped.write(raw)
    summary={k:v for k,v in r.items() if k not in ('square_diagnostics','selective_risk_coverage')}
    write(output/'fusion_reliability_d3b_summary_v1.json',summary)
    for name in ('boards','matched_coverage','board_probes'):shutil.copyfile(analysis/(name+'.csv'),output/('fusion_reliability_d3b_'+name+'.csv'))
    shutil.copytree(figures,output/'fusion_reliability_d3b_examples')
    shutil.copyfile(functional/'evaluation.json',output/'fusion_reliability_d3b_functional_v1.json')
    manifest=dict(schema_version='fusion_reliability_export_d3b_v1',uncompressed_result_sha256=hashlib.sha256(raw).hexdigest(),
                  artifacts={str(p.relative_to(output)):sha256_file(p) for p in sorted(output.rglob('*')) if p.is_file()})
    write(output/'fusion_reliability_d3b_export_v1.json',manifest)
    print('Exported reliability curves, case studies, tables and development-only functional results.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('analysis','figures','functional','output-dir'):p.add_argument('--'+flag,type=Path,required=True)
    a=p.parse_args();export(a.analysis,a.figures,a.functional,a.output_dir)
