"""Read-only public excerpt verification. Does not reconstruct omitted probability bytes."""
import argparse
import gzip
import hashlib
import json
import math
import struct
from pathlib import Path


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def load_record(root,path):
    raw=(root/path).read_bytes()
    return json.loads(gzip.decompress(raw) if path.endswith('.gz') else raw)


def verify(root):
    root=Path(root).resolve();manifest=json.loads((root/'evidence-manifest.json').read_text())
    errors=[];checked=0;raw_records=0;rows_checked=0;counts_checked=0;omitted=0
    specifications={}
    for plan_name in ['plan.json','extension-plan.json','confirmation-plan.json']:
        if not (root/plan_name).exists(): continue
        plan=json.loads((root/plan_name).read_text())
        for filename,key in [('mnist.py','scorer_sha256'),('run_modal.py','runner_sha256')]:
            if digest((root/'canonical'/filename).read_bytes())!=plan[key]: errors.append('Canonical identity mismatch: '+plan_name)
        for arm in plan['arms']: specifications[(plan['phase'],arm['name'])]=arm
    bundles={}
    for entry in manifest['files']:
        if 'manifest' in Path(entry['public_path']).name and entry['public_path'].endswith('.json'):
            value=json.loads((root/entry['public_path']).read_text())
            if value.get('format')=='sutro-d6-input-manifest-v1': bundles.update(value['entries'])
    paired={}
    for entry in manifest['files']:
        path=root/entry['public_path']
        if not path.is_file():
            errors.append('Missing public file: '+entry['public_path']);continue
        raw=path.read_bytes()
        if digest(raw)!=entry['public_sha256'] or len(raw)!=entry['public_bytes']:
            errors.append('Public file hash/size mismatch: '+entry['public_path']);continue
        checked+=1
        if 'excerpt_sha256' not in entry:
            if entry.get('representation')=='byte-identical copy' and digest(raw)!=entry['original_sha256']:
                errors.append('Original copied bytes differ: '+entry['public_path'])
            continue
        derived=gzip.decompress(raw)
        if digest(derived)!=entry['excerpt_sha256'] or len(derived)!=entry['excerpt_bytes']:
            errors.append('Derived excerpt hash/size mismatch: '+entry['public_path']);continue
        record=json.loads(derived);raw_records+=1
        marker=record.get('public_excerpt',{})
        if marker.get('full_raw_sha256')!=entry['original_sha256'] or marker.get('full_raw_bytes')!=entry['original_bytes']:
            errors.append('Original-raw provenance marker mismatch: '+entry['public_path'])
        name=record.get('arm','').split('__shard')[0]
        specification=specifications.get((record.get('phase'),name))
        if not specification or record.get('parameters')!=specification or record.get('source_sha256')!=specification['source_sha256']:
            errors.append('Raw frozen arm/parameters/source identity mismatch: '+entry['public_path'])
        source=(specification or {}).get('source_path','sources/'+name+'.py')
        if source and (root/source).exists() and digest((root/source).read_bytes())!=record.get('source_sha256'):
            errors.append('Source mismatch: '+entry['public_path'])
        actual_omitted=0
        for row in record.get('rows',[]):
            rows_checked+=1;truth=row.get('test_labels',[])
            if len(truth)!=10000 or any(type(x)is not int or x<0 or x>9 for x in truth):
                errors.append('Invalid query labels: '+entry['public_path']);continue
            if record.get('phase')!='development':
                expected=bundles.get('mnist:'+str(row.get('draw_seed')))
                if not expected or row.get('data_bundle_key')!=expected['key'] or row.get('data_bundle_npz_sha256')!=expected['npz_sha256'] or row.get('array_sha256')!={k:v['sha256']for k,v in expected['arrays'].items()} or row.get('train_sha256')!=expected['train_sha256'] or row.get('query_sha256')!=expected['query_sha256']:
                    errors.append('Frozen input manifest provenance mismatch: '+entry['public_path'])
                elif digest(struct.pack('<10000q',*truth))!=expected['arrays']['qy']['sha256']:
                    errors.append('Saved truth differs from frozen manifest: '+entry['public_path'])
                key=(record.get('phase'),row.get('draw_seed'));signature=(row.get('train_sha256'),row.get('query_sha256'),tuple(truth))
                if key in paired and paired[key]!=signature: errors.append('Strict phase coordinate/label pairing mismatch: '+entry['public_path'])
                paired[key]=signature
            for item in row.get('members',[])+row.get('prefixes',[]):
                pred=item.get('prediction',[])
                valid=len(pred)==10000 and all(type(x)is int and 0<=x<10 for x in pred)
                correct=sum(a==b for a,b in zip(pred,truth))
                if not valid or item.get('correct')!=correct or item.get('total')!=10000:
                    errors.append('Saved label/count mismatch: '+entry['public_path'])
                if 'accuracy_pct' in item and not math.isclose(item['accuracy_pct'],correct/100,rel_tol=1e-12,abs_tol=1e-9): errors.append('Saved accuracy mismatch: '+entry['public_path'])
                counts_checked+=1
            for member in row.get('members',[]):
                if any(k in member for k in ('probabilities','probabilities_f32_zlib_base64')):
                    errors.append('Unexpected public probability payload: '+entry['public_path'])
                if 'probability_payload_omitted' in member:
                    if member['probability_payload_omitted'].get('omitted')is not True:
                        errors.append('Invalid omission marker: '+entry['public_path'])
                    actual_omitted+=1
            for prefix in row.get('prefixes',[]):
                k=prefix.get('members');times=[m.get('full_call_seconds') for m in row.get('members',[])[:k]]
                if len(times)!=k or not all(isinstance(t,(float,int))and math.isfinite(t)and t>0 for t in times) or not math.isclose(sum(times),prefix.get('full_call_seconds',-1),rel_tol=1e-12,abs_tol=1e-8):
                    errors.append('Prefix fit-time sum mismatch: '+entry['public_path'])
                if 'cumulative_algorithm_seconds' in prefix:
                    algorithm=[m.get('algorithm_seconds') for m in row['members'][:k]]
                    if not all(isinstance(t,(float,int))and math.isfinite(t)and t>0 for t in algorithm) or not math.isclose(sum(algorithm),prefix['cumulative_algorithm_seconds'],rel_tol=1e-12,abs_tol=1e-7):
                        errors.append('Prefix algorithm-time sum mismatch: '+entry['public_path'])
        if actual_omitted!=marker.get('omitted_probability_payloads') or actual_omitted!=entry['omitted_probability_payloads']:
            errors.append('Omitted payload count mismatch: '+entry['public_path'])
        omitted+=actual_omitted
    planned=sum(len(json.loads((root/name).read_text())['arms'])for name in ['plan.json','extension-plan.json'])
    public_development=sum(1 for e in manifest['files']if e.get('original_relative_path','').startswith(('development/','development-extension/')))
    if manifest['status']=='final' and (planned!=19 or public_development!=19):
        errors.append('Final public package does not retain every development outcome')
    return {'status':'pass'if not errors else'fail','public_package_phase':manifest['status'],
            'files_checked':checked,'raw_excerpts_checked':raw_records,'draw_rows_checked':rows_checked,
            'saved_count_vectors_replayed':counts_checked,'explicitly_omitted_probability_payloads':omitted,
            'errors':errors,'scope':'Public hashes, original-raw provenance markers, saved labels/counts and timing sums only. Full probability and NPZ verification requires retained local evidence; not executed here.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args();result=verify(args.root);print(json.dumps(result,indent=2));raise SystemExit(0 if result['status']=='pass' else 1)
