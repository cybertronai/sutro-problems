"""CPU-only fixtures: temporary files and inert Modal CLI substitutes, never remote work."""
import importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
P=Path(__file__).with_name('launch_frozen_review.py')
spec=importlib.util.spec_from_file_location('review_guard',P);g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
class GuardTests(unittest.TestCase):
 def test_wrapped_initialized_owned_only(self):
  text='✓ Initialized. View run at \nhttps://modal.com/apps/example/main/ap-NEW123\n✓ App completed. View run at \nhttps://modal.com/apps/example/main/ap-NEW123'
  self.assertEqual(g.owned_ids(text,['ap-OLD']),['ap-NEW123']);self.assertEqual(g.owned_ids(text,['ap-NEW123']),[])
 def test_untrusted_arbitrary_url_not_owned(self):
  self.assertEqual(g.owned_ids('random https://modal.com/apps/example/main/ap-OTHER',[]),[])
 def test_zero_strict(self):
  self.assertTrue(g.zero({'State':'stopped','Tasks':0}));self.assertFalse(g.zero({'State':'running','Tasks':0}));self.assertFalse(g.zero({'State':'stopped','Tasks':1}));self.assertFalse(g.zero({'State':'stopped','Tasks':False}));self.assertFalse(g.zero({'State':'stopped'}))
 def test_atomic_write(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t)/'x.json';g.write(p,{'a':1});g.write(p,{'a':2});self.assertEqual(g.read(p),{'a':2});self.assertEqual(list(Path(t).glob('*.tmp')),[])
 def test_watch_stops_only_logged_new_app(self):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);out=root/'pr142/modal';out.mkdir(parents=True);g.write(out/'apps-before.private.json',[{'App ID':'ap-OLD'}]);g.write(out/'controller-status.json',{'started_epoch':100,'controller_pid':123,'stop_requested':True});(out/'controller.log').write_text('Initialized. View run at \nhttps://modal.com/apps/example/main/ap-NEW\n')
   lists=[[{'App ID':'ap-OLD','State':'running','Tasks':5},{'App ID':'ap-NEW','State':'running','Tasks':3}],[{'App ID':'ap-NEW','State':'stopped','Tasks':0}]]
   class Ret:returncode=0;stderr=''
   with patch.object(g,'ROOT',root),patch.object(g,'listed',side_effect=lists),patch.object(g,'cli',return_value=Ret())as cli,patch.object(g.time,'time',return_value=101),patch.object(g.time,'sleep'):
    self.assertEqual(g.watch(142),0);self.assertEqual(cli.call_args.args[0],['app','stop','ap-NEW']);self.assertEqual(g.read(out/'app-provenance.json')['status'],'PASS')
 def test_missing_id_does_not_claim_shutdown(self):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);out=root/'pr142/modal';out.mkdir(parents=True);g.write(out/'apps-before.private.json',[]);g.write(out/'controller-status.json',{'started_epoch':100,'controller_pid':123,'stop_requested':True});(out/'controller.log').write_text('failed before initialized')
   with patch.object(g,'ROOT',root),patch.object(g,'cli')as cli,patch.object(g.time,'time',return_value=101):
    self.assertEqual(g.watch(142),1);cli.assert_not_called();self.assertFalse(g.read(out/'app-provenance.json')['all_owned_apps_stopped_zero'])
 def test_frozen_command_and_reject_mutations(self):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);(root/'canonical').mkdir();source=root/'source.py';source.write_text('SOURCE')
   frozen={}
   for path in g.FROZEN:
    q=root/path;q.write_text(path);frozen[path]=g.sha(q)
   inp={str(root/k):v for k,v in frozen.items()};inp[str(P.resolve())]=g.sha(P);inp[str(source)]=g.sha(source)
   command=[str(root/'controller-env/bin/python'),'-u','canonical/run_modal.py',str(source)+':hybrid_d4','--difficulty','4','--runs','3','--json',str(root/'pr142/modal')]
   p={'pr':142,'required_runs':3,'energy_enabled':True,'stock_runner':True,'source_gate_passed':True,'source_frozen_before_launch':True,'all_outcomes_retained':True,'failed_runs_will_not_be_replaced':True,'difficulty':4,'reviewed_head':'1'*40,'input_sha256':inp,'source_path':str(source),'source_sha256':g.sha(source),'function':'hybrid_d4','command':command}
   with patch.object(g,'ROOT',root),patch.object(g,'FROZEN',frozen):
    self.assertEqual(g.validate(142,p),command)
    for key,value in [('source_gate_passed',False),('required_runs',4),('energy_enabled',False),('reviewed_head','bad'),('source_sha256','0'*64),('command',command+['--no-energy'])]:
     bad=dict(p);bad[key]=value
     with self.assertRaises(AssertionError,msg=key):g.validate(142,bad)
    source.write_text('CHANGED')
    with self.assertRaises(AssertionError):g.validate(142,p)
if __name__=='__main__':unittest.main()
