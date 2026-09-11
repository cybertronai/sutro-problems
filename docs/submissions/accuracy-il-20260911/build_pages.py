"""Render static experiment reports for the repository's existing GitHub Pages."""
import argparse
import html
import importlib.util
from pathlib import Path
import shutil
import markdown

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]/'submissions'/'1nn-v4-20260911'/'build_pages.py'
spec = importlib.util.spec_from_file_location('submission_style', BASE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def render(source, current):
    md = markdown.Markdown(extensions=['tables','fenced_code','toc'],
                           extension_configs={'toc': {'toc_depth':'2'}})
    content = md.convert(source)
    labels = [('index.html','Results'),('il.html','Intermediate language'),
              ('protocol.html','Study protocol'),('ambiguities.html','Ambiguities'),
              ('session.html','Session export')]
    nav = ''.join(f'<a class="{"active" if file == current else ""}" href="{file}">{label}</a>'
                  for file,label in labels)
    title = next(line[2:] for line in source.splitlines() if line.startswith('# '))
    style = base.STYLE + '\n@media(max-width:600px){header .brand{font-size:22px}th,td{padding:7px}table{font-size:12px}}'
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="MNIST-small higher-accuracy experiments and compact affine-loop scoring, with uniform ps and fJ units.">
<title>{html.escape(title)} · Sutro problems</title><style>{style}</style></head>
<body><header><p>Sutro problems · Feasibility study · September 2026</p>
<div class="brand">MNIST-small / Higher accuracy &amp; compact scoring</div><nav>{nav}</nav></header>
<main>{content}</main><footer>Exploratory results; MLP A100 measurements remain outstanding.
<a href="https://github.com/cybertronai/sutro-problems/tree/main/mnist/experiments/accuracy-il-20260911">Source and evidence</a>
 · <a href="../1nn-v4-20260911/">Original submission</a></footer></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('docs/submissions/accuracy-il-20260911'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for source in sorted(HERE.iterdir()):
        if source.is_file() and source.suffix in {'.py','.md','.txt','.json','.npy'}:
            shutil.copy2(source, args.output/source.name)
    for name, target in [('report.md','index.html'),('IL.md','il.html'),
                         ('study-protocol.md','protocol.html'),('ambiguities.md','ambiguities.html'),
                         ('session.md','session.html')]:
        (args.output/target).write_text(render((HERE/name).read_text(), target))
    print(args.output)


if __name__ == '__main__':
    main()
