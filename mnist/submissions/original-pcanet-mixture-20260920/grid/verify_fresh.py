"""Run the frozen full-memory checker on a newly generated execution directory."""
from pathlib import Path
import argparse,sys,tempfile
import verify_candidate as frozen
HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--execution-dir',type=Path,required=True)
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-module',type=Path,default=HERE.parent)
    p.add_argument('--a100-predictions',type=Path)
    a=p.parse_args()
    # Preserve the frozen verifier's expected evidence layout without replacing
    # any archived result. Only path resolution changes; source bytes stay fixed.
    with tempfile.TemporaryDirectory(prefix='pcanet-grid-verification-') as tmp:
        root=Path(tmp)
        for path in HERE.iterdir():
            if path.name=='full-execution':continue
            (root/path.name).symlink_to(path.resolve(),target_is_directory=path.is_dir())
        (root/'full-execution').symlink_to(a.execution_dir.resolve(),target_is_directory=True)
        frozen.HERE=root
        sys.argv=['verify_candidate.py','--raw',str(a.raw.resolve()),'--data-module',
                  str(a.data_module.resolve()),'--output',str(a.output.resolve())]
        if a.a100_predictions:sys.argv+=['--a100-predictions',str(a.a100_predictions.resolve())]
        frozen.main()


if __name__=='__main__':main()
