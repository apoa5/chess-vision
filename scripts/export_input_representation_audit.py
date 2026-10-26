"""Publish a compact C.1 review packet without copying videos or tensor archives."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil


def export(audit_dir, analysis_dir, annotations_path, output):
    if output.exists():
        raise ValueError('Use a fresh output directory')
    audit_bytes = (audit_dir / 'audit.json').read_bytes()
    audit = json.loads(audit_bytes)
    annotations = json.loads(annotations_path.read_text())
    analysis = json.loads((analysis_dir / 'analysis.json').read_text())
    digest = hashlib.sha256(audit_bytes).hexdigest()
    if annotations['audit_sha256'] != digest or analysis['audit_sha256'] != digest:
        raise ValueError('Review inputs refer to different audits')
    output.mkdir(parents=True)
    for source in audit_dir.glob('*.jpg'):
        shutil.copyfile(source, output / source.name)
    shutil.copyfile(audit_dir / 'audit.json', output / 'audit.json')
    shutil.copyfile(analysis_dir / 'analysis.json', output / 'analysis.json')
    shutil.copyfile(annotations_path, output / 'visible_extents.json')
    page = (audit_dir / 'index.html').read_text()
    page = page.replace('<h1>C.1 input representation audit</h1>',
                        '<h1>C.1 input representation audit</h1><p><a href="visible_extents.html">Visible-extent annotations and caveats</a> · <a href="../input_representation_c1.md">Report</a></p>')
    (output / 'index.html').write_text(page)
    sections = []
    for case in audit['cases']:
        cid = case['case_id']
        annotation = annotations['cases'].get(cid)
        if not annotation:
            continue
        x1, y1, x2, y2 = annotation['visible_bbox_local_xyxy']
        cx, cy = annotation['crown_local_xy']
        bx1, by1, bx2, by2 = case['native_annotation_window_xyxy']
        width, height = bx2 - bx1, by2 - by1
        overlay = f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" fill="none" stroke="orange" stroke-width="2"/><circle cx="{cx}" cy="{cy}" r="4" fill="cyan"/>'
        if 'base_local_xy' in annotation:
            px, py = annotation['base_local_xy']
            overlay += f'<circle cx="{px}" cy="{py}" r="4" fill="magenta"/>'
        sections.append(f'<section><h2>{html.escape(cid)}</h2><p>{html.escape(annotation["note"])}</p><svg width="{width}" height="{height}" viewBox="0 0 {width} {height}"><image href="{cid}_raw.jpg" width="{width}" height="{height}"/>{overlay}</svg></section>')
    (output / 'visible_extents.html').write_text('<!doctype html><meta charset="utf-8"><title>C1 visible extents</title><style>body{font:16px system-ui;max-width:1000px;margin:auto}section{border-top:1px solid #aaa}svg{max-width:100%;height:auto}</style><h1>Assistant-estimated visible extents</h1><p><a href="index.html">Full representation audit</a>. Orange: approximate visible bounding rectangle; cyan: crown/head point; magenta: uncertain base contact point. Green source outline: fitted target square. These are diagnostic estimates (approximately ±5 source pixels), not human-verified segmentation, training labels, or proof of full piece visibility. Bounding rectangles include background. Hidden silhouettes and contact points remain uncertain. Preserve existing FENs and geometry-review status.</p>' + ''.join(sections))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir', type=Path, required=True)
    parser.add_argument('--analysis-dir', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    export(args.audit_dir, args.analysis_dir, args.annotations, args.output_dir)
