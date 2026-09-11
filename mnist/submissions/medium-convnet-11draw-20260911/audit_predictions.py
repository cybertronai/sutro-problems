"""Audit all frozen eleven-draw artifacts without opening any query labels."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def ahash(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
def read(path):return json.loads(path.read_text())


def audit(root):
    protocol=read(root/'protocol.json');master=read(root/'draw_manifest.json')
    frozen=read(root/'prediction_manifest.json')
    assert frozen['protocol_sha256']==master['protocol_sha256']==sha(root/'protocol.json')
    assert frozen['draw_manifest_sha256']==sha(root/'draw_manifest.json')
    assert frozen['selected_inference']=='ensemble' and frozen['member_seeds']==[101,102,103]
    assert frozen['test_labels_opened'] is False and frozen['total_predictions']==66000
    assert len(frozen['predictions'])==len(master['draws'])==frozen['total_draws']==11
    assert protocol['dataset_seeds']==list(range(20261001,20261012))
    for name,digest in protocol['source_sha256'].items():assert sha(Path(__file__).parent/name)==digest
    config=protocol['config'];c=config['width'];d=config['depth'];h=config['head_width']
    count=9*c+(d-1)*9*c*c+2*c*d+c*81*h+h+h*10+10
    records=[]
    for i,entry in enumerate(frozen['predictions']):
        assert entry['draw_index']==i and entry['dataset_seed']==protocol['dataset_seeds'][i]
        manifest_entry=master['draws'][i]
        assert manifest_entry['draw_index']==i and sha(root/manifest_entry['path'])==manifest_entry['sha256']
        draw=read(root/manifest_entry['path']);result=read(root/entry['result_path'])
        assert result['draw_index']==i and result['dataset_seed']==entry['dataset_seed']
        assert result['manifest_sha256']==manifest_entry['sha256']
        assert result['protocol_sha256']==frozen['protocol_sha256']
        assert result['test_labels_opened'] is False
        assert sha(root/entry['result_path'])==entry['result_sha256']
        assert sha(root/entry['path'])==entry['sha256']==result['predictions_file_sha256']
        assert sha(root/entry['logits_path'])==entry['logits_sha256']==result['logits_file_sha256']
        pred=np.load(root/entry['path'],allow_pickle=False)
        assert pred.shape==(6000,) and pred.dtype==np.int64 and np.all((pred>=0)&(pred<=9))
        assert ahash(pred)==entry['array_sha256']==result['predictions_sha256']
        members={member['seed']:member for member in result['members']}
        assert sorted(members)==[101,102,103]
        with np.load(root/entry['logits_path'],allow_pickle=False) as archive:
            assert set(archive.files)=={'seed101','seed102','seed103','ensemble',
                'predictions_seed101','predictions_seed102','predictions_seed103'}
            values=[]
            for seed in [101,102,103]:
                member=members[seed];logits=archive[f'seed{seed}'];mpred=archive[f'predictions_seed{seed}']
                assert logits.shape==(6000,10) and logits.dtype==np.float32 and np.isfinite(logits).all()
                assert np.array_equal(mpred,logits.argmax(1)) and mpred.dtype==np.int64
                assert ahash(logits)==member['logits_sha256'] and ahash(mpred)==member['predictions_sha256']
                assert member['config']==config and member['epochs']==protocol['epochs']==71
                assert member['schedule_epochs']==protocol['schedule_epochs']==100
                assert member['parameter_count']==count==1404618
                assert member['training_examples']==member['query_examples']==6000
                prov=member['provenance']
                assert prov['draw_index']==i and prov['dataset_seed']==entry['dataset_seed']
                assert prov['manifest_sha256']==manifest_entry['sha256']
                assert prov['source_sha256']==protocol['source_sha256']
                assert prov['protocol_sha256']==frozen['protocol_sha256']
                assert prov['input_sha256']=={name:spec['sha256'] for name,spec in draw['arrays'].items()}
                assert prov['checkpoint_inputs']==[]
                assert [r['epoch'] for r in member['history']]==list(range(1,72))
                for row in member['history']:
                    assert row['train']['total']==6000 and 0<=row['train']['correct']<=6000
                    assert np.isfinite(row['train']['loss']) and row['train']['loss']>=0
                    expected=.001*(.02+.98*(1+np.cos(np.pi*(row['epoch']-1)/100))/2)
                    assert np.isclose(row['learning_rate'],expected,rtol=1e-12,atol=1e-15)
                if i==0:assert sha(root/member['checkpoint_file'])==member['checkpoint_sha256']
                assert member['determinism']=={'algorithms':True,'cudnn_benchmark':False,
                    'cudnn_deterministic':True,'tf32_matmul':False,'tf32_cudnn':False}
                values.append(logits)
            ensemble=np.mean(np.stack(values),axis=0,dtype=np.float64)
            assert np.array_equal(ensemble,archive['ensemble']) and archive['ensemble'].dtype==np.float64
            assert ahash(ensemble)==result['ensemble_logits_sha256']
            assert np.array_equal(pred,ensemble.argmax(1))
        assert datetime.fromisoformat(result['completed_at_utc'])<datetime.fromisoformat(frozen['frozen_at_utc'])
        records.append({'draw_index':i,'dataset_seed':entry['dataset_seed'],'passed':True})
    output={'passed':True,'checked_at_utc':datetime.now(timezone.utc).isoformat(),
        'protocol_sha256':sha(root/'protocol.json'),'prediction_manifest_sha256':sha(root/'prediction_manifest.json'),
        'members_checked':33,'histories_checked':33,'ensembles_recomputed':11,'draws':records,
        'checkpoint_scope':'Byte hashes checked for draw00; checkpoint tensors not unpickled',
        'parameter_count_per_model':count,'test_labels_opened':False,'audit_source_sha256':sha(Path(__file__))}
    (root/'prediction_audit.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--study',type=Path,required=True)
    audit(parser.parse_args().study)
