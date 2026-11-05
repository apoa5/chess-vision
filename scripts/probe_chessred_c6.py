"""C6 bounded local retrieval. Images/archive GETs are explicitly prohibited."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.training.chessred_access_c6 import run_probe

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=run_probe(a.output,'local execution environment')
    print(json.dumps(result,indent=2))
