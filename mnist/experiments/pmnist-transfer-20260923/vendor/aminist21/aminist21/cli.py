import argparse
import json
from pathlib import Path
from .common import ROOT, DATASETS, read


def main():
    parser=argparse.ArgumentParser(description="Aminist 21 Validation — fifteen tasks, one training procedure")
    sub=parser.add_subparsers(dest="command",required=True)
    for command in ("fetch","verify-data","run","score","list"):
        p=sub.add_parser(command)
        p.add_argument("--data-dir",type=Path,default=ROOT/"data")
        if command in ("fetch","verify-data","run"):
            p.add_argument("--datasets",nargs="+",choices=DATASETS,default=DATASETS)
        if command in ("run","score"):
            p.add_argument("--output",type=Path,required=True)
        if command=="run":
            p.add_argument("--adapter",default="aminist21.baselines:train_predict",help="module:function or /absolute/adapter.py:function")
            p.add_argument("--config",type=Path,help="JSON config for a custom adapter")
            p.add_argument("--recipe",default="linear-sgd")
            p.add_argument("--device",choices=["cpu","cuda"],default="cpu")
            p.add_argument("--draws",type=int,nargs="+",choices=range(11),default=list(range(11)))
    args=parser.parse_args()
    if args.command=="list":
        print("\n".join(DATASETS));return
    if args.command=="fetch":
        from .data import fetch
        fetch(args.data_dir,args.datasets);return
    if args.command=="verify-data":
        from .data import load_pool
        for name in args.datasets:
            pool=load_pool(name,args.data_dir);print(name,len(pool["labels"]),"verified")
        return
    from .suite import run_local,score
    if args.command=="score":
        result=score(args.output,args.data_dir)
    else:
        config=read(args.config) if args.config else {"recipe":args.recipe,"seed":11,"device":args.device}
        names=[name for name in DATASETS if name in args.datasets]
        draws=sorted(set(args.draws))
        result=run_local(args.adapter,config,args.output,args.data_dir,names,draws)
    print(json.dumps({"complete_v1_suite":result["complete_v1_suite"],"dataset_order":result["dataset_order"],"fifteen_numbers":result["fifteen_numbers"]},indent=2))
