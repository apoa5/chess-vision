#!/usr/bin/env python3
"""Explicitly download official OCR weights into ignored tools/ocr/models."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.scanning.learned_markings import ROOT
sys.path.insert(0,str(ROOT/'tools/ocr/python'))
import easyocr
import hashlib,json

def main():
    easyocr.Reader(['en'],gpu=False,model_storage_directory=str(ROOT/'tools/ocr/models'),user_network_directory=str(ROOT/'tools/ocr/network'),verbose=True)
    weights={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'tools/ocr/models').glob('*.pth')}
    (ROOT/'tools/ocr/models/weights.json').write_text(json.dumps({'reader':'easyocr_1.7.2','sha256':weights},indent=2)+'\n')
    print('OCR weights ready; inference downloads are disabled.')

if __name__=='__main__':main()
