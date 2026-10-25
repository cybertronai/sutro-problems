"""Run one stock canonical group per predeclared candidate; preserve every outcome."""
from pathlib import Path
import datetime,hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parent
role=sys.argv[1]; planpath=ROOT/'pr140'/role/'measurement-plan.json';plan=json.loads(planpath.read_text())
for path,expected in plan['input_sha256'].items():
 assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
assert not (ROOT/'pr140'/role/'modal/controller-status.json').exists(),'Never relaunch an existing group'
out=ROOT/'pr140'/role/'modal';out.mkdir(parents=True)
status={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':plan['command'],'running':True,'measurement_plan_sha256':hashlib.sha256(planpath.read_bytes()).hexdigest()}
(out/'controller-status.json').write_text(json.dumps(status,indent=2)+'\n')
with (out/'controller.log').open('xb') as f:
 result=subprocess.run(plan['command'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
status.update(running=False,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),returncode=result.returncode)
(out/'controller-status.json').write_text(json.dumps(status,indent=2)+'\n')
sys.exit(result.returncode)
