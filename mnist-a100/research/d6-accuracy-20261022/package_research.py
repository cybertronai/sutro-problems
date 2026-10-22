"""Explicit-allowlist D6 evidence packaging. Local output only; no Git or GPU actions."""
import argparse
import ast
import base64
import datetime
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile

from research_report import render

ROOT = Path(__file__).resolve().parent
MAX_PUBLIC_BYTES = 50_000_000
TARGET_COMPRESSED_BYTES = 25_000_000
# Each optional name is deliberate. Additional reviewed artifacts must be named
# with --extra-file; no wildcard or directory traversal enters the package.
STATIC = [
 'plan.json','extension-plan.json','extension-plan-original9x4.json',
 'extension-plan-before-input-bundles.json','extension-development-context.json',
 'extension-source-audit.json','extension-kernel-coverage.json','extension-final-review.json',
 'extension-READY.json','extension-root-launch-freeze.json',
 'inputs-development-manifest.json','inputs-confirmation-manifest.json',
 'confirmation-input-manifest.json','confirmation-plan.json',
 'confirmation-frozen-development-analysis.json','confirmation-root-launch-freeze.json',
 'analysis.json','provisional-table.json','confirmation-analysis.json','confirmation-table.json',
 'controller.log','controller-status.json','extension-controller.log','extension-controller-status.json',
 'confirmation-controller.log','confirmation-controller-status.json','prepare-inputs-controller.log',
 'extension-infrastructure-events.json','confirmation-infrastructure-events.json',
 'extension-call-graph.json','confirmation-call-graph.json',
 'extension-app-status.json','confirmation-app-status.json',
 'prepare-confirmation-inputs-controller.log','bootstrap-failure.log','bootstrap-failure-status.json',
 'serialization-failure.log','serialization-failure-remote.log','serialization-failure-status.json',
 'serialization-failure-call-graph.json','independent-study-review.json',
 'shared-input-local-tests.json','independent-shared-input-review.json',
 'independent-confirmation-review.json','independent-final-research-review.json',
 'independent-control-audit.json','independent-analyzer-fixtures.json','independent_analyzer_tests.py',
 'independent-confirmation-fixtures.json','independent_confirmation_tests.py',
 'maintainer-interpretation.json','analysis-controller-readiness.json',
 'independent-frozen-confirmation-jobs.json',
 'study.py','study-before-input-bundles.py','study-extension-snapshot.py',
 'study-confirmation-snapshot.py','input_bundles.py','input-bundles-extension-snapshot.py',
 'input-bundles-confirmation-snapshot.py','confirm.py','analyze.py','analyze_confirmation.py',
 'freeze_confirmation.py','freeze_extension.py','verify_extension.py','patch_study_inputs.py',
 'parent-source.py','canonical/mnist.py','canonical/run_modal.py',
 'plot_results.py','plot_confirmation.py','development-frontier.png','development-frontier.pdf',
 'confirmation-frontier.png','confirmation-frontier.pdf','confirmation-results.png',
 'confirmation-results.pdf','regularization-effect.png','regularization-effect.pdf',
 'publication-policy.json','provenance.json','package_research.py','research_report.py','verify_public_evidence.py'
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def inside(root, path):
    resolved = (root/path).resolve()
    assert resolved.is_relative_to(root.resolve()), f'Outside evidence root: {path}'
    assert not resolved.is_symlink()
    return resolved


def gzip_bytes(raw):
    stream = io.BytesIO()
    with gzip.GzipFile(fileobj=stream, mode='wb', filename='', mtime=0) as handle:
        handle.write(raw)
    return stream.getvalue()


def excerpt(raw, original_path):
    full = json.loads(raw)
    removed = 0
    for row in full.get('rows', []):
        for member in row.get('members', []):
            payload = member.pop('probabilities', None)
            legacy = member.pop('probabilities_f32_zlib_base64', None)
            codec = member.pop('probabilities_codec', None)
            shape = member.pop('probabilities_shape', None)
            prob_sha = member.pop('probabilities_sha256', None)
            if payload is not None or legacy is not None:
                actual = payload if payload is not None else legacy
                marker = {'omitted':True,'retained_in':'Original local raw JSON and complete local evidence archive',
                          'codec':codec or ('zlib-base64-f32-le' if legacy is not None else 'nested-numeric-lists'),
                          'shape':shape,'uncompressed_FP32_sha256':prob_sha}
                if isinstance(actual,str):
                    marker['compressed_probability_payload_sha256'] = sha(base64.b64decode(actual,validate=True))
                member['probability_payload_omitted'] = marker
                removed += 1
    full['public_excerpt'] = {'format':'sutro-d6-result-excerpt-v1',
       'original_relative_path':original_path,'full_raw_sha256':sha(raw),'full_raw_bytes':len(raw),
       'omitted_probability_payloads':removed,
       'scope':'Derived excerpt, not original raw bytes. Labels, counts, timing and provenance remain. Omitted probabilities cannot be reconstructed from this public file.'}
    return (json.dumps(full,separators=(',',':'),ensure_ascii=False)+'\n').encode(), removed


def planned_files(root):
    files, records, roles = set(), [], []
    for plan_name, directory in [('plan.json','development'),('extension-plan.json','development-extension')]:
        plan = read_json(root/plan_name)
        for arm in plan['arms']:
            source = arm.get('source_path','sources/'+arm['name']+'.py')
            assert sha(inside(root,source).read_bytes()) == arm['source_sha256']
            files.add(source)
            records.append(directory+'/'+arm['name']+'.json')
    confirmation_path = root/'confirmation-plan.json'
    if confirmation_path.exists():
        plan = read_json(confirmation_path)
        for arm in plan['arms']:
            assert sha(inside(root,arm['source_path']).read_bytes()) == arm['source_sha256']
            files.add(arm['source_path'])
            for index in range(1,len(plan['shards'])+1):
                records.append(f"confirmation/{arm['name']}__shard{index}.json")
            roles.append({'arm':arm['name'],'roles':arm.get('roles',[arm['name']]),'draw_seeds':plan['draw_seeds']})
    return files, records, roles


def readiness(root, records, roles):
    problems = []
    analysis = read_json(root/'analysis.json')
    if analysis.get('all_development_finalized') is not True or analysis.get('finalized_logical_arms') != 19:
        problems.append('All nineteen development outcomes must be terminal before final conclusions')
    if analysis.get('selection_locked') or analysis.get('selected') is None:
        problems.append('Development selection is not frozen')
    for arm in analysis.get('arms',[]):
        path=root/arm['raw_path']
        if path.exists() and file_sha(path)!=arm.get('raw_sha256'):
            problems.append('Development audit is stale for '+arm['raw_path'])
    if not roles:
        problems.append('Frozen fresh confirmation plan is absent')
    for spec in roles:
        seen = []
        plan = read_json(root/'confirmation-plan.json')
        for index, seeds in enumerate(plan['shards'],1):
            path = root/f"confirmation/{spec['arm']}__shard{index}.json"
            if not path.exists():
                problems.append('Missing confirmation shard: '+path.name)
                continue
            raw = read_json(path)
            if raw.get('completed') is not True:
                problems.append('Failed/incomplete confirmation shard: '+path.name)
            actual = [row['draw_seed'] for row in raw.get('rows',[])]
            if actual != seeds:
                problems.append('Confirmation shard draw set differs: '+path.name)
            seen.extend(actual)
        if seen != spec['draw_seeds'] or len(seen) != 11:
            problems.append('Not all eleven frozen fresh draws are retained for '+spec['arm'])
    if not (root/'confirmation-analysis.json').exists():
        problems.append('Completed local full-probability confirmation audit is absent')
    else:
        audit=read_json(root/'confirmation-analysis.json')
        if audit.get('all_shards_finalized') is not True or audit.get('all_arms_valid') is not True or audit.get('global_errors'):
            problems.append('Local full-probability confirmation audit is incomplete or invalid')
        if (root/'confirmation-plan.json').exists() and file_sha(root/'confirmation-plan.json')!=audit.get('plan_sha256'):
            problems.append('Confirmation audit references a different frozen plan')
        for arm in audit.get('arms',[]):
            for shard in arm.get('shards',[]):
                path=root/shard['path']
                if not path.exists() or file_sha(path)!=shard.get('raw_sha256'):
                    problems.append('Confirmation audit is stale for '+shard['path'])
    for name in records:
        if not (root/name).exists():
            problems.append('Missing planned raw result: '+name)
    return problems


def package(root, destination, archive, final=False, extras=()):
    root, destination, archive = Path(root).resolve(), Path(destination).resolve(), Path(archive).resolve()
    assert destination.is_relative_to(root) and archive.is_relative_to(root), 'This helper writes only under the original output directory'
    assert not destination.exists(), 'Refuse to replace an existing package snapshot'
    assert not archive.exists(), 'Refuse to replace a complete evidence archive'
    sources, records, roles = planned_files(root)
    problems = readiness(root,records,roles)
    if final:
        assert not problems, '\n'.join(problems)
    files = {name for name in STATIC if (root/name).is_file()} | sources
    for name in extras:
        path = inside(root,name)
        assert path.is_file() and not name.startswith('inputs/'), 'Public extra must be a reviewed non-input file'
        files.add(str(path.relative_to(root)))
    for name in ['plan.json','extension-plan.json','analysis.json','canonical/mnist.py','canonical/run_modal.py','verify_public_evidence.py']:
        assert name in files, 'Missing mandatory package file: '+name
    # Include every declared manifest, even when a confirmation CLI supplied it
    # outside the immutable winner plan. Only explicit manifest candidates enter.
    local_inputs = set()
    for name in files:
        if 'manifest' in Path(name).name and name.endswith('.json'):
            value = read_json(root/name)
            if value.get('format') == 'sutro-d6-input-manifest-v1':
                for input_path in value.get('local_archives',{}).values():
                    local_inputs.add(str(inside(root,input_path).relative_to(root)))
    destination.mkdir(parents=True)
    inventory = []
    for name in sorted(files):
        source = inside(root,name); raw = source.read_bytes()
        target = destination/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        assert len(raw)<MAX_PUBLIC_BYTES, 'Public file above50MB: '+name
        inventory.append({'public_path':name,'public_sha256':sha(raw),'public_bytes':len(raw),
                          'original_relative_path':name,'original_sha256':sha(raw),'original_bytes':len(raw),
                          'representation':'byte-identical copy'})
    for name in records:
        path = root/name
        if not path.exists():
            continue
        raw = path.read_bytes();derived, removed = excerpt(raw,name);compressed = gzip_bytes(derived)
        target_name = name+'.gz';target=destination/target_name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(compressed)
        assert len(compressed)<MAX_PUBLIC_BYTES
        inventory.append({'public_path':target_name,'public_sha256':sha(compressed),'public_bytes':len(compressed),
                          'original_relative_path':name,'original_sha256':sha(raw),'original_bytes':len(raw),
                          'excerpt_sha256':sha(derived),'excerpt_bytes':len(derived),'omitted_probability_payloads':removed,
                          'representation':'deterministic gzip of explicitly marked derived excerpt'})
    readme = render(root,final=final).encode()
    (destination/'README.md').write_bytes(readme)
    inventory.append({'public_path':'README.md','public_sha256':sha(readme),'public_bytes':len(readme),
                      'representation':'report generated from retained analysis'})
    # The complete local archive contains original records and released-array
    # NPZs, not public excerpts. Explicit names only; no recursive directory walk.
    local_only = {'telegram-discussion.json'} if (root/'telegram-discussion.json').exists() else set()
    originals = set(files) | {name for name in records if (root/name).exists()} | local_inputs | local_only
    local_inventory = []
    local_inventory_path = root/('full-local-inventory-final.json' if final else 'full-local-inventory-draft.json')
    public_originals={entry['original_relative_path']:entry['original_sha256'] for entry in inventory if 'original_relative_path' in entry}
    # Read and hash the exact bytes handed to tar; do not let a concurrent edit
    # turn the inventory into a claim about earlier, different file contents.
    with tarfile.open(archive,'w:gz') as handle:
        for name in sorted(originals):
            raw=(root/name).read_bytes();digest=sha(raw)
            if name in public_originals:
                assert digest==public_originals[name], 'Evidence changed during packaging: '+name
            local_inventory.append({'path':name,'sha256':digest,'bytes':len(raw)})
            info=tarfile.TarInfo(name);info.size=len(raw);info.mtime=0;info.mode=0o644
            handle.addfile(info,io.BytesIO(raw))
        local_inventory_bytes=(json.dumps({'format':'sutro-d6-full-local-inventory-v1','files':local_inventory,
            'includes_full_probability_payloads':True,'includes_released_array_NPZs':True,
            'contains_pixels_or_secret_release_maps':False},indent=2)+'\n').encode()
        local_inventory_path.write_bytes(local_inventory_bytes)
        info=tarfile.TarInfo('full-local-inventory.json');info.size=len(local_inventory_bytes);info.mtime=0;info.mode=0o644
        handle.addfile(info,io.BytesIO(local_inventory_bytes))
    archive_sha = file_sha(archive)
    source_scope=[]
    if (root/'confirmation-plan.json').exists():
        for arm in read_json(root/'confirmation-plan.json')['arms']:
            tree=ast.parse((root/arm['source_path']).read_text())
            source_scope.append({'roles':arm['roles'],'source_path':arm['source_path'],'source_sha256':arm['source_sha256'],
                                 'source_bytes':arm['source_bytes'],'official_size_cap_bytes':20480,
                                 'fits_official_source_size_cap':arm['source_bytes']<=20480,
                                 'research_top_level_functions':[node.name for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))],
                                 'official_label_entrypoint_or_qualification_claimed':False})
    report = {'format':'sutro-d6-public-evidence-manifest-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'status':'final' if final else 'draft','final_readiness_problems':problems,
              'intended_repository_path':'mnist-a100/research/d6-accuracy-20261022','files':inventory,
              'source_raw_representation':'Public raw result paths are gzip excerpts; original payload bytes are retained locally, not published.',
              'all_declared_development_arms':19,'public_development_outcomes':sum(1 for f in inventory if f.get('original_relative_path','').startswith(('development/','development-extension/'))),
              'complete_local_archive':{'local_path':str(archive),'sha256':archive_sha,'bytes':archive.stat().st_size,
                                        'inventory_sha256':sha(local_inventory_bytes),'publicly_distributed_in_repository':False},
              'private_local_inputs':[{'path':name,'sha256':sha((root/name).read_bytes()),'bytes':(root/name).stat().st_size} for name in sorted(local_inputs)],
              'scope':'Public verification checks labels/counts/timing and published hashes only. Original probability and NPZ reconstruction requires local archive and its retained audit.',
              'confirmation_source_scope':source_scope,
              'no_official_score_energy_or_global_accuracy_ceiling_claim':True}
    (destination/'evidence-manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    # Estimate repository transport size with one deterministic gzip of each
    # uncompressed copied file. Already-gzip files are counted as stored.
    compressed_total = sum(p.stat().st_size if p.suffix=='.gz' else len(gzip_bytes(p.read_bytes())) for p in destination.rglob('*') if p.is_file())
    assert compressed_total<TARGET_COMPRESSED_BYTES, f'Public compressed package above25MB: {compressed_total}'
    result={'package':str(destination),'status':report['status'],'public_compressed_size_estimate':compressed_total,
            'complete_local_archive':str(archive),'archive_sha256':archive_sha,'readiness_problems':problems,
            'all_package_files_below50MB':True,'source_raw_bytes_copied_identically_except_declared_excerpts':True}
    (root/('publication-package-final.json' if final else 'publication-package-draft.json')).write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--final',action='store_true')
    parser.add_argument('--extra-file',action='append',default=[])
    args=parser.parse_args()
    print(json.dumps(package(args.root,args.destination,args.archive,args.final,args.extra_file),indent=2))
