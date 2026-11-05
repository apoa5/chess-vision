"""C6 EU CPU-only access test. No GPU, model dependencies, or training entrypoint."""
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import modal

app=modal.App('chess-vision-c6-readiness')
image=modal.Image.debian_slim(python_version='3.12').add_local_file('src/training/chessred_access_c6.py','/root/chessred_access_c6.py')
data=modal.Volume.from_name('chess-vision-c4-data-v1')

@app.function(image=image,cpu=(.25,.25),memory=(512,2048),region='eu',nonpreemptible=True,
              timeout=180,startup_timeout=120,retries=0,max_containers=1,min_containers=0,
              scaledown_window=2,volumes={'/data':data})
def probe(run_id):
    from chessred_access_c6 import run_probe
    result=run_probe(Path('/data')/run_id,'Modal CPU, EU region');data.commit();return result

@app.local_entrypoint()
def main(run_id:str='chessred-c6-eu-v1'):
    workspace=modal.Workspace.from_context();workspace.hydrate()
    if workspace.name!='akwasipoku-oforiagyei':raise ValueError('Unexpected workspace')
    output=Path('outputs/evaluation/chessred_c6')/run_id
    if output.exists():raise ValueError('Fresh local output required')
    result=probe.remote(run_id);output.mkdir(parents=True)
    # Only bounded metadata/annotations, never image archives, are copied locally.
    for record in result['records']:
        for attempt in record['attempts']:
            if 'path' not in attempt:continue
            remote=attempt['path'];dest=output/Path(remote).name;sha=hashlib.sha256();size=0
            with dest.open('xb') as f:
                for chunk in data.read_file(remote):
                    size+=len(chunk)
                    if size>96*1024*1024:raise ValueError('Export byte limit exceeded')
                    sha.update(chunk);f.write(chunk)
            if sha.hexdigest()!=attempt['sha256']:raise ValueError('Metadata export checksum mismatch')
            attempt['local_path']=str(dest)
    (output/'retrieval.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
