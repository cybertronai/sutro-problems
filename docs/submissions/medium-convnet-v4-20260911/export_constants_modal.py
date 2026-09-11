"""Reproduce the seed-only compiler constants in the pinned CPU environment."""
from pathlib import Path
import json
import modal

HERE = Path(__file__).resolve().parent
IMAGE_REF = ('ghcr.io/ab-10/wikitext-bench@'
 'sha256:95de89319ba89c53a91d5440a5db4ff46f68b05031062c0a760ec3caa48dc42f')
image = modal.Image.from_registry(IMAGE_REF).pip_install('numpy==2.2.6')
for name in ('schedule.py', 'export_constants.py'):
    image = image.add_local_file(str(HERE/'ordered_backend'/name), '/root/constants_source/'+name)
image = image.add_local_file(str(HERE/'config.json'), '/root/config.json')
app = modal.App('sutro-mnist-medium-seed-constants')

@app.function(image=image, cpu=2, memory=4096, timeout=300)
def export():
    import subprocess
    subprocess.run(['python', '/root/constants_source/export_constants.py',
        '--config', '/root/config.json', '--seeds', '11', '22', '33',
        '--n-train', '10000', '--output', '/tmp/exported', '--include-initial'], check=True)
    return {p.name:p.read_bytes() for p in Path('/tmp/exported').iterdir()}

@app.local_entrypoint()
def main():
    out = HERE/'constants'
    out.mkdir(exist_ok=True)
    assert not any(out.iterdir()), 'Refusing to overwrite constants'
    for name, value in export.remote().items():
        assert Path(name).name == name
        (out/name).write_bytes(value)
    print(json.dumps({'output':str(out), 'scope':'CPU seed generation; no dataset mounted'}))
