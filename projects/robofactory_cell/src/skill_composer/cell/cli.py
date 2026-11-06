import argparse,json
from pathlib import Path
from .dataset import RAW,DATA,RUN
from .program import Node
from ..common import write_json

TASK_NAMES=['relay_stack','relay_container','handoff','parallel_stack','container']


def main():
    parser=argparse.ArgumentParser(description='Multi-task RoboFactory skill composition: published recordings only')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('download',help='Download/verify 750 published episodes across five tasks (~5.6 GB)');p.add_argument('--raw',type=Path,default=RAW)
    p=sub.add_parser('prepare',help='Annotate published recordings and split by source episode');p.add_argument('--raw',type=Path,default=RAW);p.add_argument('--output',type=Path,default=DATA)
    p=sub.add_parser('train',help='Behavior-clone a graph-conditioned coordinator');p.add_argument('--data',type=Path,default=DATA);p.add_argument('--run',type=Path,default=RUN);p.add_argument('--epochs',type=int,default=200)
    p=sub.add_parser('evaluate',help='Held-out imitation metrics on original published episodes');p.add_argument('--data',type=Path,default=DATA);p.add_argument('--run',type=Path,default=RUN)
    p=sub.add_parser('rollout',help='Watch learned composition execute an unseen multi-part task');p.add_argument('--run',type=Path,default=RUN);p.add_argument('--task',choices=TASK_NAMES,default='relay_container');p.add_argument('--parts',type=int,choices=[2,3,4],default=3);p.add_argument('--controller',choices=['learned','serial','greedy','arm0','arm1'],default='learned');p.add_argument('--seed',type=int,default=0);p.add_argument('--viewer',action='store_true');p.add_argument('--gif',type=Path);p.add_argument('--delay',type=float,default=0.)
    p=sub.add_parser('skills',help='Inspect each reusable frozen skill, independently of the learned head');p.add_argument('--run',type=Path,default=RUN);p.add_argument('--arm',type=int,choices=[0,1],default=0);p.add_argument('--skill',choices=['pick','place','stack','open_lid','all'],default='all');p.add_argument('--viewer',action='store_true');p.add_argument('--gif',type=Path)
    p=sub.add_parser('compare',help='Closed-loop evaluation against serial and dependency-aware greedy coordinators');p.add_argument('--run',type=Path,default=RUN);p.add_argument('--seeds',type=int,default=20);p.add_argument('--seed-start',type=int,default=1000);p.add_argument('--parts',type=int,choices=[2,3,4],default=3)
    p=sub.add_parser('preview',help='Preview original published demonstration images with inferred operation boundaries');p.add_argument('--run',type=Path,default=RUN);p.add_argument('--task',choices=['PickMeat','StackCube','PassShoe','PlaceFood','TwoRobotsStackCube'],default='PassShoe');p.add_argument('--episode',type=int,default=0)
    p=sub.add_parser('demo',help='Render reproducible individual-skill and side-by-side coordination galleries');p.add_argument('--run',type=Path,default=RUN)
    args=parser.parse_args()
    if args.command=='download':
        from .dataset import download
        download(args.raw)
    elif args.command=='prepare':
        from .dataset import prepare
        prepare(args.raw,args.output)
    elif args.command=='train':
        from .policy import train
        train(args.data,args.run,args.epochs)
    elif args.command=='evaluate':
        from .policy import evaluate
        evaluate(args.data,args.run)
    elif args.command=='rollout':
        from .experiment import episode
        r=episode(args.task,args.parts,args.controller,args.seed,args.run,args.viewer,args.gif,delay=args.delay)
        write_json(args.run/'rollout.json',r)
        print(json.dumps({k:v for k,v in r.items() if k!='dispatch_trace'},indent=2))
    elif args.command=='skills':
        from .experiment import episode
        all_skills=['pick','place','stack','open_lid'] if args.skill=='all' else [args.skill]
        for skill in all_skills:
            if skill=='open_lid':nodes=[Node('open_lid',args.arm,obj='lid')]
            else:
                nodes=[Node('pick',args.arm)]
                if skill in ('place','stack'):nodes.append(Node('place',args.arm,deps=(0,),stack=skill=='stack'))
            gif=args.gif
            if gif and len(all_skills)>1:gif=gif.with_name(gif.stem+'_'+skill+gif.suffix)
            r=episode('individual_'+skill,controller='greedy',run=args.run,viewer=args.viewer,gif=gif,nodes=nodes)
            write_json(args.run/f'arm{args.arm}_{skill}.json',r)
            print(f'Arm {args.arm} frozen {skill}: success={r["success"]} time={r["time_s"]}s')
    elif args.command=='compare':
        from .experiment import evaluate
        evaluate(args.run,args.seeds,args.parts,args.seed_start)
    elif args.command=='preview':
        from .preview import preview
        preview(args.task,args.episode,args.run)
    elif args.command=='demo':
        from .visuals import demos
        demos(args.run)
