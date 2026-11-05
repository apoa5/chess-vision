"""Bounded official-source access probe; no image archive or training operation."""
import concurrent.futures,hashlib,json,socket,time,urllib.error,urllib.parse,urllib.request
from pathlib import Path

DATASET='99b5c721-280b-450b-b058-b2900b69a90f'
ANNOTATION_URL=f'https://data.4tu.nl/file/{DATASET}/3cae6364-daca-4967-b426-1e4b68cdb64c'
IMAGES_URL=f'https://data.4tu.nl/file/{DATASET}/6329e969-616e-48e3-b893-a0379d1c15ba'
ANNOTATION_MD5='d34bca5ad46ec7a8df96a1d3c36784f3'
REQUESTS=(
 ('dataset_metadata',f'https://data.4tu.nl/v2/datasets/{DATASET}','GET',2*1024*1024),
 ('file_inventory',f'https://data.4tu.nl/v2/datasets/{DATASET}/files','GET',2*1024*1024),
 ('annotation_metadata',ANNOTATION_URL,'HEAD',0),
 ('annotations',ANNOTATION_URL,'GET',96*1024*1024),
 ('image_archive_metadata',IMAGES_URL,'HEAD',0),
)

def bounded_fetch(name,url,method,limit,output,*,timeout=12,attempts=2,opener=None):
    """Bound requests/bytes, check publisher MD5, and never retry whole archives."""
    if method not in ('GET','HEAD') or not 1<=attempts<=2 or not 0<timeout<=20:raise ValueError('Invalid probe limits')
    if urllib.parse.urlsplit(url).scheme!='https' or urllib.parse.urlsplit(url).hostname!='data.4tu.nl':raise ValueError('Official HTTPS endpoint required')
    if urllib.parse.urlsplit(url).path==urllib.parse.urlsplit(IMAGES_URL).path and method!='HEAD':raise ValueError('Full image-archive download forbidden')
    if not 0<=limit<=96*1024*1024:raise ValueError('Excessive download limit')
    output=Path(output);opener=opener or urllib.request.urlopen;history=[]
    for attempt in range(attempts):
        start=time.monotonic();record=dict(attempt=attempt+1)
        try:
            request=urllib.request.Request(url,method=method,headers={'User-Agent':'Chess-Vision C6 official readiness probe','Accept':'application/json,application/octet-stream;q=0.9,*/*;q=0.1'})
            with opener(request,timeout=timeout) as response:
                headers=response.headers;length=headers.get('Content-Length')
                record.update(status=response.status,content_type=headers.get('Content-Type'),content_length=int(length) if length and length.isdigit() else None,
                              etag=headers.get('ETag'),last_modified=headers.get('Last-Modified'))
                if method=='HEAD':
                    record.update(bytes=0,elapsed_seconds=time.monotonic()-start);history.append(record);return dict(name=name,url=url,method=method,attempts=history,success=True)
                if response.status!=200:raise ValueError('Full HTTP200 response required for verification')
                if record['content_length'] is not None and record['content_length']>limit:raise ValueError('Published length exceeds bounded budget')
                if 'zip' in (record['content_type'] or '').lower():raise ValueError('Archive response refused')
                dest=output/(name+'.response');tmp=dest.with_suffix('.part')
                if dest.exists() or tmp.exists():raise ValueError('Fresh response path required')
                sha=hashlib.sha256();md5=hashlib.md5();total=0
                with tmp.open('xb') as f:
                    while True:
                        if time.monotonic()-start>60:raise TimeoutError('Overall response deadline exceeded')
                        chunk=response.read(min(65536,limit-total+1))
                        if not chunk:break
                        total+=len(chunk)
                        if total>limit:raise ValueError('Streaming byte limit exceeded')
                        sha.update(chunk);md5.update(chunk);f.write(chunk)
                if record['content_length'] is not None and total!=record['content_length']:raise ValueError('Truncated response')
                tmp.replace(dest);record.update(bytes=total,sha256=sha.hexdigest(),md5=md5.hexdigest(),path=str(dest))
                if name=='annotations':
                    # Parse here only to distinguish an annotation object from an HTML error page.
                    with dest.open() as f:annotation=json.load(f)
                    if not isinstance(annotation,dict) or not {'images','categories','annotations','splits'}<=set(annotation):raise ValueError('Not the documented annotation object')
                    record['published_md5_verified']=md5.hexdigest()==ANNOTATION_MD5
                    if not record['published_md5_verified']:raise ValueError('Published annotation MD5 mismatch; quarantine')
                record['elapsed_seconds']=time.monotonic()-start;history.append(record)
                return dict(name=name,url=url,method=method,attempts=history,success=True)
        except (OSError,ValueError,urllib.error.HTTPError) as e:
            record.update(error=type(e).__name__+': '+str(e),elapsed_seconds=time.monotonic()-start);history.append(record)
            # Retry only a transport failure that wrote no partial artifact.
            if isinstance(e,urllib.error.HTTPError) or not isinstance(e,(TimeoutError,urllib.error.URLError)) or any(output.glob(name+'.*')):break
            if attempt+1<attempts:time.sleep(.5)
    return dict(name=name,url=url,method=method,attempts=history,success=False)

def run_probe(output,location):
    output=Path(output)
    if output.exists():raise ValueError('Fresh versioned probe directory required')
    output.mkdir(parents=True);begin=time.monotonic()
    try:addresses=sorted({r[4][0] for r in socket.getaddrinfo('data.4tu.nl',443,type=socket.SOCK_STREAM)})
    except OSError as e:addresses=dict(error=type(e).__name__+': '+str(e))
    def fetch(row):return bounded_fetch(*row,output)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:records=list(pool.map(fetch,REQUESTS))
    result=dict(schema_version='chessred_official_access_c6_v1',location=location,dns_addresses=addresses,records=records,
                elapsed_seconds=time.monotonic()-begin,images_downloaded=0,full_archive_downloaded=False,training_started=False)
    (output/'retrieval.json').write_text(json.dumps(result,indent=2)+'\n');return result
