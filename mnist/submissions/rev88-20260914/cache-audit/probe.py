"""Read-only A100 profiler/counter capability probe."""
from pathlib import Path
import json
import modal
IMAGE_REF='ghcr.io/ab-10/wikitext-bench@sha256:95de89319ba89c53a91d5440a5db4ff46f68b05031062c0a760ec3caa48dc42f'
image=modal.Image.from_registry(IMAGE_REF).apt_install('gcc').pip_install('numpy==2.2.6','nvidia-ml-py==12.560.30')
app=modal.App('sutro-rev88-cache-probe-20260914')
@app.function(image=image,gpu='A100-40GB',cpu=2,memory=8192,timeout=300,min_containers=0,max_containers=1,scaledown_window=2)
def probe():
    import subprocess,glob,os
    commands=[['bash','-c','command -v ncu || true; command -v nvcc || true'],
              ['bash','-c','find /opt/nvidia /usr/local/cuda* -name ncu -type f 2>/dev/null | head -12'],
              ['bash','-c','cat /proc/driver/nvidia/params 2>/dev/null | head -50'],
              ['bash','-c','apt-cache search nsight-compute | head -12'],
              ['nvidia-smi','--query-gpu=name,uuid,memory.total','--format=csv,noheader']]
    results=[]
    for command in commands:
        r=subprocess.run(command,capture_output=True,text=True,timeout=30)
        results.append({'command':command,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
    return {'results':results,'uid':os.getuid()}
@app.local_entrypoint()
def main():
    r=probe.remote();p=Path(__file__).parent/'probe.json';p.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
