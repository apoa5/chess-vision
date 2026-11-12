"""Small VP9 browser fixture from registered approved HD recording samples.

This is a derived 1fps replay, not an original continuous recording or new data.
No original source or cached frame is modified. Only the compact video persists.
"""
from pathlib import Path
import argparse,hashlib,json,subprocess,os
import cv2
ROOT=Path('outputs/evaluation/assisted_mvp16_v1')
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--kind',choices=('replay','single','mp4-alias'),default='replay')
 args=parser.parse_args()
 if args.kind=='mp4-alias':
  media=json.loads(Path('data/metadata/a3_pilot_01_sources_v2.json').read_text())['sessions'][0]['compressed_comparison_source']
  original=Path(media['path'])
  if hashlib.sha256(original.read_bytes()).hexdigest()!=media['sha256']:raise ValueError('Source hash changed')
  if media['codec']!='h264' or media['container_metadata']['major_brand']!='mp42':raise ValueError('MP4 alias requires verified MP4/H264 bytes')
  alias=ROOT/'approved-starting.mp4'
  if alias.exists() or alias.is_symlink():raise ValueError('Alias already exists')
  alias.symlink_to(os.path.relpath(original.resolve(),ROOT.resolve()))
  print(alias)
  return
 source=json.loads(Path('outputs/evaluation/assisted_mvp15_v1/replay_source_v1.json').read_text())
 single=args.kind=='single';frames=[source['frames'][7]] if single else source['frames'][:12];output=ROOT/('single_view_2s.webm' if single else 'hd_replay_12s_720.webm')
 if output.exists():raise ValueError('Fixture already exists')
 ffmpeg='outputs/evaluation/assisted_mvp1_v1/browsers/ffmpeg-1013/ffmpeg-linux'
 cmd=[ffmpeg,'-hide_banner','-loglevel','error','-f','image2pipe','-framerate','1/2' if single else '1','-vcodec','mjpeg','-i','pipe:0','-an','-c:v','libvpx-vp9','-threads','1','-b:v','700k',str(output)]
 with subprocess.Popen(cmd,stdin=subprocess.PIPE) as process:
  for frame in frames:
   raw=Path(frame['path']).read_bytes()
   if hashlib.sha256(raw).hexdigest()!=frame['derivative_sha256']:raise ValueError('Frozen HD sample changed')
   image=cv2.imread(frame['path']);image=cv2.resize(image,(406,720),interpolation=cv2.INTER_AREA)
   ok,jpeg=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,95])
   if not ok:raise ValueError('Frame encoding failed')
   process.stdin.write(jpeg.tobytes())
  process.stdin.close()
  if process.wait(timeout=60):raise RuntimeError('Fixture encoding failed')
 manifest=dict(kind='derived_functional_fixture',path=str(output),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),bytes=output.stat().st_size,width=406,height=720,duration_seconds=2 if single else 12,source=source,frames=frames,mapping='one registered source sample shown for 2 seconds' if single else 'one registered source sample per second; changed timeline, not original recording')
 (ROOT/('single_view_fixture_v1.json' if single else 'hd_replay_fixture_v1.json')).write_text(json.dumps(manifest,indent=2));print(output,output.stat().st_size)
if __name__=='__main__':main()
