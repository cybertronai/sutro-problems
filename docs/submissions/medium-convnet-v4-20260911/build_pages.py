"""Build standalone submission pages from an explicit public-evidence allowlist.

Requires Markdown3.x. The output is local until its commit reaches main.
Raw datasets, log files, caches and the source session JSONL are never copied.
"""
import argparse
import hashlib
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
from urllib.parse import unquote, urlsplit
import markdown

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_OUTPUT = REPO / 'docs/submissions/medium-convnet-v4-20260911'
STYLE = '\n:root{color-scheme:light;--ink:#18242e;--muted:#5b6a73;--line:#d6e0e4;--accent:#066d77}\n*{box-sizing:border-box}body{margin:0;background:#f5f7f7;color:var(--ink);font:17px/1.65 system-ui,-apple-system,sans-serif}\nheader{background:#103c47;color:white;padding:36px max(24px,calc((100vw - 1060px)/2)) 26px}\nheader p{margin:0;font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:#bfd9da}\nheader .brand{font-size:27px;font-weight:680;margin:6px 0 18px}nav{display:flex;gap:9px;flex-wrap:wrap}\nnav a{color:#edf8f8;border:1px solid #5c8088;padding:6px 13px;border-radius:5px;text-decoration:none;font-size:14px}\nnav a.active{background:#fff;color:#103c47}main{max-width:1120px;margin:0 auto;padding:32px 30px 70px;background:white;min-height:60vh}\nh1{font:700 36px/1.2 Georgia,serif;letter-spacing:-.015em;margin:10px 0 24px}h2{font-size:24px;line-height:1.35;margin:38px 0 16px;border-top:1px solid var(--line);padding-top:24px}\nh3{font-size:19px;margin:28px 0 8px}p{margin:12px 0}a{color:var(--accent);text-underline-offset:3px;overflow-wrap:anywhere}\ncode{font-size:.84em;background:#edf2f3;padding:2px 5px;border-radius:3px;overflow-wrap:anywhere}pre{padding:18px;overflow:auto;background:#edf2f3;border:1px solid var(--line);border-radius:6px;font-size:14px;line-height:1.5}\npre code{padding:0;background:none;overflow-wrap:normal}table{border-collapse:collapse;display:block;max-width:100%;overflow:auto;margin:22px 0;font-size:15px}th,td{padding:10px 14px;border:1px solid var(--line);vertical-align:top}th{background:#eaf2f3;text-align:left}tr:nth-child(even) td{background:#f8fafb}\nblockquote{border-left:4px solid #dfa83d;margin:20px 0;padding:10px 20px;background:#fffaed}blockquote p{margin:6px 0}li{margin:9px 0}footer{max-width:1120px;margin:auto;padding:22px 30px 40px;font-size:13px;color:var(--muted)}.toc{padding:18px 24px;background:#f4f8f8;border:1px solid var(--line);border-radius:6px}.toc ul{margin:0;padding-left:20px}.toc li{margin:3px 0}.toc a{text-decoration:none}\n@media(max-width:600px){main{padding:24px 18px 44px}h1{font-size:29px}header{padding:24px 18px}table{font-size:13px}th,td{padding:8px}}\n'

# Positive allowlist. Adding evidence requires reviewing a filename or category.
ROOT_FILES = {
    'README.md', 'report.md', 'ambiguities.md', 'session.md', 'session.json',
    'build_pages.py', 'build_report.py', 'export_session.py', 'requirements-report.txt', 'audit.py',
    'ensemble.py', 'evaluation.py', 'export_constants_modal.py', 'measure_gpu.py',
    'modal_run.py', 'reproduce.py', 'test_evaluation.py', 'capture_argmax_modal.py',
    'ir_bn.py', 'ir_bn_reference.py', 'ir_conv.py', 'ir_core.py', 'ir_dense.py',
    'ir_machine.py', 'ir_model.py', 'ir_score.py', 'ir_tables.py',
    'ir_test_bn.py', 'ir_test_conv.py', 'ir_test_core.py', 'ir_test_model.py',
    'accuracy.json', 'accuracy_run_plan.json', 'config.json', 'draw_manifest.json',
    'evaluator-audit.json', 'ir-bn-validation.json', 'ir-conv-validation.json',
    'ir-core-validation.json', 'ir-model-validation.json', 'model-score.json',
    'prediction_manifest.json', 'program.il.json', 'protocol.json', 'selection.json',
    'audit.json', 'submission-audit.json', 'reproduction-audit.json', 'reproduction-check.json',
}
BACKEND_SOURCES = {
    'README.md', 'ops.py', 'network.py', 'schedule.py', 'cpu_ref.py',
    'bn_ops.py', 'bn_ref.py', 'export_constants.py', 'check_network.py',
    'check_submission.py', 'check_primitives.py', 'native_initialization_reference.py',
}
VALIDATION_DIRECTORIES = {
    'primitive-results-02', 'network-results-03',
    'network-results-bn-dropout', 'submission-validation',
}
FORBIDDEN_COMPONENTS = {'data', 'raw', 'logs', '__pycache__', '.git', 'rawsession'}


def allowed(relative):
    parts = relative.parts
    if any(part in FORBIDDEN_COMPONENTS for part in parts):
        return False
    if len(parts) == 1:
        return relative.name in ROOT_FILES
    first = parts[0]
    if first in {'draws', 'results'}:
        return len(parts) == 2 and re.fullmatch(r'draw-\d{2}\.json', relative.name) is not None
    if first == 'predictions':
        return len(parts) == 2 and re.fullmatch(r'draw-\d{2}\.npz', relative.name) is not None
    if first in {'initial', 'checkpoints'}:
        return len(parts) == 2 and relative.name in {'parameters.npz', 'draw-00.npz'}
    if first == 'constants':
        return len(parts) == 2 and (relative.name == 'constants.json' or
               re.fullmatch(r'initial-seed-\d+\.npz', relative.name) is not None)
    if first == 'ptx':
        return len(parts) == 2 and relative.suffix == '.ptx'
    if first == 'supplemental-argmax':
        if len(parts) == 2:
            return relative.name in {'results.json', 'source-freeze.json'} or relative.suffix == '.ptx'
        return len(parts) == 3 and parts[1] == 'sources' and relative.suffix == '.py'
    if first == 'benchmark':
        if len(parts) == 2:
            return relative.name in {'results.json', 'run_plan.json', 'audit.json', 'measurement_protocol.json'}
        return allowed(Path(*parts[1:])) and parts[1] in {'results', 'predictions', 'initial', 'checkpoints', 'ptx'}
    if first == 'ordered_backend':
        if len(parts) == 2:
            return relative.name in BACKEND_SOURCES
        if parts[1] not in VALIDATION_DIRECTORIES:
            return False
        if len(parts) == 3:
            return relative.name in {'results.json', 'source-freeze.json'} or relative.suffix in {'.ptx', '.npz', '.npy'}
        return len(parts) == 4 and parts[2] == 'sources' and relative.name in BACKEND_SOURCES
    return False


def render(source, current):
    md = markdown.Markdown(extensions=['tables', 'fenced_code', 'toc'],
                           extension_configs={'toc': {'toc_depth': '2'}})
    content = md.convert(source)
    if current == 'session.html':
        content = re.sub(r'<a href="(/(?:tmp|private|Users|home)/[^\"]*)"[^>]*>(.*?)</a>',
                         lambda match: '<span title="Local artifact recorded in the original session">' + match.group(2) + '</span>',
                         content, flags=re.DOTALL)
    for name, target in [('report.md', 'index.html'), ('session.md', 'session.html'),
                         ('ambiguities.md', 'ambiguities.html')]:
        content = re.sub(r'(href=[\"\'])' + re.escape(name) + r'(?=[#\"\'])',
                         lambda match: match.group(1) + target, content)
    labels = [('index.html', 'Results & reproduction'), ('session.html', 'Session export'),
              ('ambiguities.html', 'Ambiguities & problems')]
    nav = ''.join(f'<a class="{"active" if name == current else ""}" href="{name}">{label}</a>'
                  for name, label in labels)
    title = next((line[2:] for line in source.splitlines() if line.startswith('# ')), 'MNIST-medium submission')
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Reproducible MNIST-medium convolutional-network submission: eleven draws, exact model scoring and complete-task A100 measurements.">
<title>{html.escape(title)} · Sutro problems</title><style>{STYLE}</style></head>
<body><header><p>Sutro problems · Submission attempt · September 2026</p><div class="brand">MNIST-medium / Ordered ConvNet</div><nav>{nav}</nav></header>
<main>{content}</main><footer>Standalone record. The report states the evaluated threshold and dataset protocol. <a href="https://github.com/cybertronai/sutro-problems">Repository</a> · <a href="../">All submission reports</a></footer></body></html>'''


class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.links = []; self.ids = set()
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs: self.ids.add(attrs['id'])
        for key in ('href', 'src'):
            if key in attrs: self.links.append(attrs[key])


def check_links(output):
    checked = 0
    for name in ('index.html', 'ambiguities.html', 'session.html'):
        path = output / name; parser = Links(); parser.feed(path.read_text())
        for value in parser.links:
            url = urlsplit(value)
            if url.scheme or url.netloc or not url.path and not url.fragment:
                continue
            if url.path.startswith('/'):
                # Session messages may quote local workspace links. They remain
                # historical text and are not links to the public evidence bundle.
                if name == 'session.html':
                    continue
                target = REPO / unquote(url.path.lstrip('/'))
            else:
                target = path.parent / unquote(url.path) if url.path else path
            if not target.exists():
                raise ValueError(f'Broken local link in {name}: {value}')
            if url.fragment and target.suffix == '.html':
                target_parser = Links(); target_parser.feed(target.read_text())
                if unquote(url.fragment) not in target_parser.ids:
                    raise ValueError(f'Unknown local anchor in {name}: {value}')
            checked += 1
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(); output = args.output.resolve()
    assert output != HERE and HERE not in output.parents
    for filename in ('report.md', 'ambiguities.md', 'session.md'):
        if not (HERE / filename).is_file():
            raise FileNotFoundError(f'Required human-readable source is missing: {filename}')
    output.mkdir(parents=True, exist_ok=True)
    published = []
    for source in sorted(HERE.rglob('*')):
        relative = source.relative_to(HERE)
        if source.is_file() and not source.is_symlink() and allowed(relative):
            target = output / relative; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            published.append({'path': relative.as_posix(), 'bytes': source.stat().st_size,
                              'sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
    # Remove obsolete generated copies only when the previous build explicitly
    # declared ownership. Never recursively delete the output directory.
    inventory = output / 'evidence.json'
    if inventory.exists():
        previous = json.loads(inventory.read_text()).get('files', [])
        current = {row['path'] for row in published}
        for row in previous:
            rel = Path(row['path'])
            if rel.is_absolute() or '..' in rel.parts:
                raise ValueError('Unsafe previous inventory path')
            stale = output / rel
            if row['path'] not in current and stale.is_file():
                stale.unlink()
    for source, target in [('report.md', 'index.html'), ('ambiguities.md', 'ambiguities.html'),
                           ('session.md', 'session.html')]:
        (output / target).write_text(render((HERE / source).read_text(), target), encoding='utf-8')
    inventory.write_text(json.dumps({'policy': 'Explicit reviewed filenames and evidence categories only; no datasets or raw sessions.',
                                    'files': published}, indent=2) + '\n')
    checked = check_links(output)
    print(json.dumps({'output': str(output), 'copied_evidence_files': len(published),
                      'copied_bytes': sum(row['bytes'] for row in published), 'checked_local_links': checked}, indent=2))


if __name__ == '__main__':
    main()
