"""Render the saved report, audit and visible conversation as standalone HTML.

Install requirements-report.txt, then run from any directory. No remote assets.
"""
import argparse
import html
from pathlib import Path
import shutil
import markdown

STYLE = '''
:root{color-scheme:light;--ink:#18242e;--muted:#5b6a73;--line:#d6e0e4;--accent:#066d77}
*{box-sizing:border-box}body{margin:0;background:#f5f7f7;color:var(--ink);font:17px/1.65 system-ui,-apple-system,sans-serif}
header{background:#103c47;color:white;padding:36px max(24px,calc((100vw - 1060px)/2)) 26px}
header p{margin:0;font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:#bfd9da}
header .brand{font-size:27px;font-weight:680;margin:6px 0 18px}nav{display:flex;gap:9px;flex-wrap:wrap}
nav a{color:#edf8f8;border:1px solid #5c8088;padding:6px 13px;border-radius:5px;text-decoration:none;font-size:14px}
nav a.active{background:#fff;color:#103c47}main{max-width:1120px;margin:0 auto;padding:32px 30px 70px;background:white;min-height:60vh}
h1{font:700 36px/1.2 Georgia,serif;letter-spacing:-.015em;margin:10px 0 24px}h2{font-size:24px;line-height:1.35;margin:38px 0 16px;border-top:1px solid var(--line);padding-top:24px}
h3{font-size:19px;margin:28px 0 8px}p{margin:12px 0}a{color:var(--accent);text-underline-offset:3px;overflow-wrap:anywhere}
code{font-size:.84em;background:#edf2f3;padding:2px 5px;border-radius:3px;overflow-wrap:anywhere}pre{padding:18px;overflow:auto;background:#edf2f3;border:1px solid var(--line);border-radius:6px;font-size:14px;line-height:1.5}
pre code{padding:0;background:none;overflow-wrap:normal}table{border-collapse:collapse;display:block;max-width:100%;overflow:auto;margin:22px 0;font-size:15px}th,td{padding:10px 14px;border:1px solid var(--line);vertical-align:top}th{background:#eaf2f3;text-align:left}tr:nth-child(even) td{background:#f8fafb}
blockquote{border-left:4px solid #dfa83d;margin:20px 0;padding:10px 20px;background:#fffaed}blockquote p{margin:6px 0}li{margin:9px 0}footer{max-width:1120px;margin:auto;padding:22px 30px 40px;font-size:13px;color:var(--muted)}.toc{padding:18px 24px;background:#f4f8f8;border:1px solid var(--line);border-radius:6px}.toc ul{margin:0;padding-left:20px}.toc li{margin:3px 0}.toc a{text-decoration:none}
@media(max-width:600px){main{padding:24px 18px 44px}h1{font-size:29px}header{padding:24px 18px}table{font-size:13px}th,td{padding:8px}}
'''


def render(source, current):
    md = markdown.Markdown(extensions=['tables', 'fenced_code', 'toc'],
                           extension_configs={'toc': {'toc_depth': '2'}})
    content = md.convert(source)
    labels = [('index.html', 'Results & reproduction'), ('session.html', 'Session export'),
              ('ambiguities.html', 'Ambiguities & problems')]
    nav = ''.join(f'<a class="{"active" if file == current else ""}" href="{file}">{label}</a>'
                  for file, label in labels)
    title = next((line[2:] for line in source.splitlines() if line.startswith('# ')), 'MNIST-small')
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Reproducible MNIST-small 1-nearest-neighbor submission attempt, model scores and measured A100 energy.">
<title>{html.escape(title)} · Sutro problems</title><style>{STYLE}</style></head>
<body><header><p>Sutro problems · Submission attempt · September 2026</p><div class="brand">MNIST-small / 1-nearest-neighbor</div><nav>{nav}</nav></header>
<main>{content}</main><footer>Standalone record. Model scores use declared conventions; acceptance remains subject to review. <a href="https://github.com/cybertronai/sutro-problems">Repository</a> · <a href="../">All submission reports</a></footer></body></html>'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('docs/submissions/1nn-v4-20260911'))
    args = p.parse_args()
    here = Path(__file__).resolve().parent
    args.output.mkdir(parents=True, exist_ok=True)
    # Keep the small reproduction artifacts beside the reports; never copy local data/cache/logs.
    for source in sorted(here.iterdir()):
        if source.is_file() and source.suffix in {'.py', '.md', '.txt', '.json', '.npy', '.v4', '.ir', '.csv', '.u32le', '.ptx'}:
            shutil.copy2(source, args.output / source.name)
    for filename, target in [('report.md','index.html'), ('session.md','session.html'),
                              ('ambiguities.md','ambiguities.html')]:
        (args.output / target).write_text(render((here / filename).read_text(), target))
    print(args.output)


if __name__ == '__main__':
    main()
