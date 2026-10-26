"""Descriptive uncertainty distributions by correctness and true class; no tuning."""
import argparse
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.select_video_keyframes import read,write
from src.scanning.fusion_reliability import SIGNALS
from src.scanning.native_pilot import sha256_file


def summarize(analysis,output):
    r=read(analysis);groups=[]
    for method in ('single_quality','mean_probabilities','mean_logits'):
        rows=[s for s in r['square_diagnostics'] if s['method']==method]
        strata=[('all',rows),('occupied',[s for s in rows if s['occupied']]),('empty',[s for s in rows if not s['occupied']])]
        strata += [('class_'+str(k),[s for s in rows if s['true_class_id']==k]) for k in range(13)]
        for stratum,squares in strata:
            for name,predicate in [('correct',lambda s:s['correct']),('incorrect',lambda s:not s['correct']),
                                   ('confidently_incorrect',lambda s:s['confidently_wrong']),('low_confidence',lambda s:s['maximum_probability']<.8)]:
                selected=[s for s in squares if predicate(s)];metrics={}
                for signal in SIGNALS:
                    values=[s[signal] for s in selected]
                    metrics[signal]=dict(mean=statistics.mean(values),median=statistics.median(values),minimum=min(values),maximum=max(values)) if values else None
                groups.append(dict(method=method,stratum=stratum,group=name,count=len(selected),signals=metrics))
    write(output,dict(schema_version='fusion_reliability_signal_groups_d3b_v1',analysis_sha256=sha256_file(analysis),groups=groups,
        warning='Overlapping descriptive groups, not independent trials; null metrics for zero support'))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();summarize(a.analysis,a.output)
