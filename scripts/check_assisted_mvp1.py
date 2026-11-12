"""Verify immutable research/media inputs, recording authorized app-only edits."""
import argparse
import hashlib
import json
from pathlib import Path

ALLOWED_APPLICATION_EDITS = {
    'scripts/serve_scan_web.py', 'web/src/main.tsx', 'web/src/DigitalBoard.tsx',
    'web/src/style.css', 'web/package.json', 'web/dist/index.html',
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def check_snapshot(expected, allowed):
    changes, missing = [], []
    unchanged = 0
    for name, before in expected.items():
        if 'holdout' in name.lower():
            raise ValueError('Holdout path access prohibited')
        path = Path(name)
        if not path.is_file():
            missing.append(name)
            continue
        after = sha256(path)
        if after == before:
            unchanged += 1
        else:
            changes.append(dict(path=name, before_sha256=before, after_sha256=after,
                                authorized_application_edit=name in allowed))
    return dict(schema_version='assisted_mvp1_preservation_v1', checked=len(expected),
                unchanged=unchanged, authorized_application_changes=[c for c in changes if c['authorized_application_edit']],
                unexpected_changes=[c for c in changes if not c['authorized_application_edit']],
                missing=missing, final_holdout_access=False,
                checkpoints_and_original_media_unchanged=not missing and not any(not c['authorized_application_edit'] for c in changes))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', type=Path, default=Path('outputs/evaluation/assisted_mvp1_v1/preserved_inputs_v1.json'))
    p.add_argument('--output', type=Path, default=Path('outputs/evaluation/assisted_mvp1_v1/integrity_v1.json'))
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Preservation output already exists; choose a fresh output filename')
    baseline = json.loads(args.snapshot.read_text())
    if set(baseline['authorized_application_edits']) != ALLOWED_APPLICATION_EDITS:
        raise ValueError('The application-only exception list differs from the authorized MVP scope')
    result = check_snapshot(baseline['sha256'], ALLOWED_APPLICATION_EDITS)
    with args.output.open('x') as output:
        output.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
    if result['unexpected_changes'] or result['missing']:
        raise ValueError('Protected files differ; inspect the preservation report')


if __name__ == '__main__':
    main()
