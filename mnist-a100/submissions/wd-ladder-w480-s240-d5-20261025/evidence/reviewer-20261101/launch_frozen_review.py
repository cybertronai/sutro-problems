"""Finite stock-Modal review launcher. Only root invokes run; watch is an independent local stopper."""
from pathlib import Path
import argparse,datetime,fcntl,hashlib,json,os,re,signal,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
LIMIT_S=1200
STOP_AT_S=1140
APP_RE=re.compile(r'(?:Initialized\. View run at|App completed\. View run at)\s+https://modal\.com/apps/[^/\s]+/[^/\s]+/(ap-[A-Za-z0-9]+)')
FROZEN={'canonical/mnist.py':'1bc6d8d96a776565bc70f3ccdb91be7de57187d4495b2b0437fac98f5e120064','canonical/run_modal.py':'1180b843cf6c32414774da90c835a8b7298d2db8719989237a793ca1c62a10ab','saved_record_audit.py':'88647522871d80423f37ffae29e457b9511cf7fcc9d0f18351d334f8ff1ca2fc'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
 p=Path(p);q=p.with_name(p.name+f'.{os.getpid()}.tmp');q.write_text(json.dumps(d,indent=2)+'\n');os.replace(q,p)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def cli(args,timeout=15):
 return subprocess.run([str(ROOT/'controller-env/bin/python'),'-m','modal',*args],capture_output=True,text=True,timeout=timeout)
def listed():
 r=cli(['app','list','--json']);assert r.returncode==0,'Cannot read Modal app inventory'
 d=json.loads(r.stdout);assert isinstance(d,list),'Unexpected app inventory';return d
def appid(row):return row.get('App ID',row.get('AppID',row.get('app_id')))
def zero(row):
 tasks=row.get('Tasks',row.get('tasks'));return str(row.get('State',row.get('state',''))).lower()=='stopped' and not isinstance(tasks,bool) and str(tasks)=='0'
def owned_ids(log,baseline):
 return sorted(set(APP_RE.findall(log))-set(baseline))
def validate(pr,plan):
 assert pr in (142,143) and plan['pr']==pr
 assert plan['required_runs']==3 and plan['energy_enabled'] is True and plan['stock_runner'] is True
 assert plan['source_gate_passed'] is True and plan['source_frozen_before_launch'] is True
 assert plan['all_outcomes_retained'] is True and plan['failed_runs_will_not_be_replaced'] is True
 assert plan['difficulty']==(4 if pr==142 else 5)
 assert re.fullmatch('[0-9a-f]{40}',plan['reviewed_head'])
 for p,h in plan['input_sha256'].items():assert sha(p)==h,'Frozen input changed: '+p
 for p,h in FROZEN.items():assert sha(ROOT/p)==h and plan['input_sha256'].get(str(ROOT/p))==h
 assert plan['input_sha256'].get(str(Path(__file__).resolve()))==sha(__file__),'Guard not pinned'
 source=Path(plan['source_path']);assert source.is_absolute() and sha(source)==plan['source_sha256']
 assert plan['input_sha256'].get(str(source))==plan['source_sha256']
 cmd=[str(ROOT/'controller-env/bin/python'),'-u','canonical/run_modal.py',str(source)+':'+plan['function'],'--difficulty',str(plan['difficulty']),'--runs','3','--json',str(ROOT/f'pr{pr}/modal')]
 assert plan['command']==cmd,'Non-stock command/options';return cmd

def watch(pr):
 out=ROOT/f'pr{pr}/modal';status=out/'controller-status.json';s=read(status);start=s['started_epoch'];baseline=read(out/'apps-before.private.json');ids=[];stops=[];error=None
 while True:
  s=read(status);ids=owned_ids((out/'controller.log').read_text(errors='replace'),[appid(r)for r in baseline]);elapsed=time.time()-start
  if elapsed>=STOP_AT_S or s.get('stop_requested') or os.getppid()!=s['controller_pid']:break
  time.sleep(1)
 # No app ID is an unconfirmed outcome, never proof of a stopped remote worker.
 if not ids:error='No owned initialized App ID captured; no remote stop may be claimed'
 deadline=start+LIMIT_S
 while ids and time.time()<deadline:
  try:
   rows=listed();by={appid(r):r for r in rows};pending=[x for x in ids if x not in by or not zero(by[x])]
   if not pending:
    proof={'status':'PASS','owned_app_ids':ids,'all_owned_apps_stopped_zero':True,'owned_rows':[by[x]for x in ids],'stop_attempts':stops,'finished_utc':now(),'elapsed_seconds':time.time()-start,'deadline_reached':time.time()-start>=STOP_AT_S}
    write(out/'app-provenance.json',proof);return 0
   for aid in pending:
    assert aid in ids and aid not in {appid(r)for r in baseline},'Refusing unowned stop'
    r=cli(['app','stop',aid],timeout=15);stops.append({'app_id':aid,'utc':now(),'returncode':r.returncode,'stderr':r.stderr[-1000:]})
  except Exception as e:error=type(e).__name__+': '+str(e)
  time.sleep(1)
 write(out/'app-provenance.json',{'status':'STOP_UNCONFIRMED','owned_app_ids':ids,'all_owned_apps_stopped_zero':False,'stop_attempts':stops,'error':error or 'Remote stop not verified before the wall deadline','finished_utc':now(),'elapsed_seconds':time.time()-start,'deadline_reached':True});return 1

def run(pr):
 lock=(ROOT/'review-launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 planpath=ROOT/f'pr{pr}/measurement-plan.json';plan=read(planpath);cmd=validate(pr,plan)
 for other in (142,143):
  p=ROOT/f'pr{other}/modal/controller-status.json'
  if p.exists():
   assert other!=pr,'Initial group already launched; never relaunch'
   prev=read(p);proof=read(p.parent/'app-provenance.json')
   assert prev['running']is False and proof['status']=='PASS' and proof['all_owned_apps_stopped_zero']is True,'Previous group not stopped'
 out=ROOT/f'pr{pr}/modal';out.mkdir(exist_ok=True)
 assert not list(out.iterdir()),'Never overwrite retained group output'
 baseline=listed();write(out/'apps-before.private.json',baseline)
 s={'pr':pr,'reviewed_head':plan['reviewed_head'],'source_sha256':plan['source_sha256'],'measurement_plan_sha256':sha(planpath),'command':cmd,'started_utc':now(),'started_epoch':time.time(),'controller_pid':os.getpid(),'running':True,'returncode':None,'deadline_reached':False,'stop_requested':False,'wall_limit_seconds':LIMIT_S,'stop_request_at_seconds':STOP_AT_S,'stock_runner_and_resources_unchanged':True,'estimated_cost_is_not_a_dollar_cap':True}
 write(out/'controller-status.json',s)
 child=None;cancel=[False]
 def interrupt(signum,frame):cancel[0]=True
 old={sig:signal.signal(sig,interrupt)for sig in (signal.SIGINT,signal.SIGTERM)}
 try:
  with (out/'controller.log').open('xb')as log,(out/'watchdog.log').open('xb')as wl:
   stopper=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'watch',str(pr)],stdout=wl,stderr=subprocess.STDOUT,start_new_session=True)
   child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   while child.poll()is None:
    elapsed=time.time()-s['started_epoch']
    if cancel[0]or elapsed>=STOP_AT_S:
     s.update(stop_requested=True,deadline_reached=elapsed>=STOP_AT_S);write(out/'controller-status.json',s)
     if elapsed>=LIMIT_S:child.kill();break
     if cancel[0]:os.killpg(child.pid,signal.SIGINT);cancel[0]=False
    time.sleep(1)
   s['returncode']=child.wait();s['stop_requested']=True;write(out/'controller-status.json',s)
   # The independent stopper is already requesting/confirming owned-app shutdown.
   stopper.wait(timeout=max(1,s['started_epoch']+LIMIT_S-time.time())+20)
 except BaseException as e:
  s['error']=type(e).__name__+': '+str(e);s['stop_requested']=True
  if child and child.poll()is None:child.terminate()
 finally:
  for sig,handler in old.items():signal.signal(sig,handler)
  s.update(running=False,finished_utc=now(),elapsed_seconds=time.time()-s['started_epoch']);s['deadline_reached']=s['deadline_reached']or s['elapsed_seconds']>=STOP_AT_S
  s['retained_run_files']=[p.name for p in sorted(out.glob('run-*.json'))];write(out/'controller-status.json',s)
 proof=read(out/'app-provenance.json')if(out/'app-provenance.json').exists()else{}
 return 0 if s['returncode']==0 and proof.get('status')=='PASS'and not s.get('error')else 1

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['run','watch']);p.add_argument('pr',type=int,choices=[142,143]);a=p.parse_args();return run(a.pr)if a.action=='run'else watch(a.pr)
if __name__=='__main__':sys.exit(main())
