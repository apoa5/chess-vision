#!/usr/bin/env python3
"""Replay a browser capture ZIP into deterministic diagnostics and manual overlays."""
import argparse
from pathlib import Path
import sys
from zipfile import BadZipFile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.scanning.replay import replay_session


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path)
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/scanning/replay'))
    args=parser.parse_args()
    try:report=replay_session(args.archive,args.output_dir)
    except (ValueError,OSError,KeyError,TypeError,BadZipFile) as exc:parser.exit(1,f'Replay failed: {exc}\n')
    print(f"Replayed {len(report['frames'])} frames; report: {args.output_dir/'replay.json'}")
    print('Recognition is not connected. No squares resolved and no automatic completion claimed.')

if __name__=='__main__':main()
