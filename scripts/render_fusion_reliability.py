"""Dependency-free standalone SVG curves and confidence tables for D.3b."""
import argparse
import csv
import html
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.select_video_keyframes import read,write
from src.scanning.native_pilot import sha256_file
from src.scanning.fusion_reliability import SIGNALS

METHODS=('single_quality','mean_probabilities','mean_logits')
COLORS=('#2864b4','#c06b00','#39843c')


def panels(path,title,series,columns=3):
    """Render discrete observed values; lines are visual guides, never interpolated metrics."""
    rows=(len(series)+columns-1)//columns;width=columns*440;height=rows*330+85
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>',
           f'<text x="20" y="23" font-size="17">{html.escape(title)}</text>']
    for n,(name,xlabel,ylabel,curves,ymax) in enumerate(series):
        ox=(n%columns)*440+62;oy=(n//columns)*330+58
        def point(x,y):return (ox+x*330,oy+230-y/ymax*230)
        parts.append(f'<text x="{ox}" y="{oy-10}" font-size="14">{html.escape(name)}</text>')
        for t in range(6):
            x=ox+t*66;y=oy+230-t*46
            parts.append(f'<path d="M {x} {oy} V {oy+230} M {ox} {y} H {ox+330}" stroke="#ddd" fill="none"/>')
            parts.append(f'<text x="{x-8}" y="{oy+249}" font-size="10">{t/5:.1f}</text><text x="{ox-40}" y="{y+4}" font-size="10">{t*ymax/5:.2f}</text>')
        parts.append(f'<text x="{ox}" y="{oy+271}" font-size="11">{html.escape(xlabel)}</text><text x="{ox}" y="{oy+288}" font-size="11">{html.escape(ylabel)}</text>')
        if name in ('all','occupied','empty'):
            parts.append(f'<path d="M {ox} {oy+230} L {ox+330} {oy}" stroke="#777" stroke-dasharray="4,4" fill="none"/>')
        for j,(label,xy) in enumerate(curves):
            color=COLORS[METHODS.index(label)];coords=[point(x,y) for x,y in xy]
            if coords:parts.append('<polyline points="'+' '.join(f'{x:.2f},{y:.2f}' for x,y in coords)+f'" fill="none" stroke="{color}" stroke-width="1.5"/>')
            parts.extend(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2" fill="{color}"/>' for x,y in coords)
    for j,label in enumerate(METHODS):
        parts.append(f'<text x="{20+j*390}" y="{height-12}" fill="{COLORS[j]}" font-size="13">{label}</text>')
    parts.append('</svg>');path.write_text(''.join(parts))


def render(analysis,output):
    if output.exists(): raise ValueError('Fresh figure directory required')
    r=read(analysis); output.mkdir(parents=True);bins=[];series=[]
    for stratum in ('all','occupied','empty'):
        curves=[]
        for method in METHODS:
            a=next(a for a in r['strata'] if a['method']==method and a['stratum']==stratum)
            points=[p for p in a['reliability']['reliability_bins'] if p['count']]
            curves.append((method,[(p['mean_score'],p['accuracy']) for p in points]))
            for p in a['reliability']['reliability_bins']:bins.append(dict(method=method,stratum=stratum,**p))
        series.append((stratum,'Mean probability (uncalibrated)','Observed accuracy',curves,1.))
    panels(output/'reliability.svg','Five correlated development sessions; nonempty bins only, no calibration fit',series)
    series=[]
    for signal in SIGNALS:
        curves=[]
        for method in METHODS:
            c=next(c for c in r['selective_risk_coverage'] if c['method']==method and c['stratum']=='occupied' and c['signal']==signal)
            curves.append((method,[(p['coverage'],p['risk']) for p in c['points'] if p['risk'] is not None]))
        series.append((signal,'True-occupied square coverage','Retained error rate',curves,1.))
    panels(output/'occupied_risk_coverage.svg','Complete-tie threshold curves; truth stratification is evaluation only',series,columns=3)
    series=[]
    for method in METHODS:
        curves=[(method,sorted([(b['mean_probability'],b['incorrect_squares']/64) for b in r['board_diagnostics'] if b['method']==method]))]
        # Every board coordinate is also in boards.csv; connecting points does not imply a model.
        series.append((method,'Mean square probability','Incorrect squares / 64 (five boards)',curves,1.))
    panels(output/'board_mean_risk.svg','High average confidence does not establish exact reconstruction; five discrete boards',series)
    with (output/'confidence_bins.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(bins[0]));w.writeheader();w.writerows(bins)
    cases=[]
    for name,position,square in [('starting_recovered','starting','g1'),('puzzle_broken','puzzle','d7'),('endgame_confident_miss','endgame','h4')]:
        cases.append(dict(name=name,position=position,canonical_square=square,methods=[s for s in r['square_diagnostics'] if s['position_id'].endswith(position) and s['canonical_square']==square]))
    write(output/'cases_v1.json',dict(schema_version='fusion_reliability_cases_d3b_v1',analysis_sha256=sha256_file(analysis),cases=cases))
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>D.3b development reliability audit</h1><p>Independent validation pending; no safe operating threshold established. Points are discrete observations; connecting lines are visual guides.</p>'+''.join(f'<h2>{p.stem}</h2><img style="max-width:100%" src="{p.name}">' for p in sorted(output.glob('*.svg')))+'<p><a href="confidence_bins.csv">Confidence-bin counts</a> · <a href="cases_v1.json">Three case studies and uncertainty values</a></p>')
    print('Rendered three SVG figures and confidence/case tables.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();render(a.analysis,a.output_dir)
