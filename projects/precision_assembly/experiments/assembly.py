"""Precision assembly: released experts, efficiency head, native paired benchmarks."""
import argparse
from contextlib import nullcontext
import json
import time
from pathlib import Path

from src.runtime import configure_caches


def main():
    configure_caches()
    settings=json.loads((Path(__file__).parent/'configs/assembly.json').read_text())
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task',choices=('list','freeze','preflight','train','baseline','evaluate','benchmark'))
    parser.add_argument('--episodes',type=int,default=settings['evaluation_episodes'])
    parser.add_argument('--seed',type=int)
    parser.add_argument('--kits',type=int,default=settings['kits'])
    parser.add_argument('--samples',type=int,default=settings['preflight_samples_per_skill'])
    parser.add_argument('--problems',type=int,default=settings['head_training_problems'])
    parser.add_argument('--epochs',type=int,default=settings['head_epochs'])
    parser.add_argument('--rounds',type=int,default=settings['dagger_rounds'])
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--checkpoint')
    parser.add_argument('--verify-only',action='store_true',help='Run qualification without replacing the trained head duration model')
    parser.add_argument('--scheduler',choices=('sequential','fifo','shortest','critical_path','johnson','lookahead'),default='critical_path')
    parser.add_argument('--proxy',action='store_true',help='Explicitly use the duration surrogate, not robot physics')
    args=parser.parse_args()
    args.seed=args.seed if args.seed is not None else (1000 if args.task=='preflight' else settings['evaluation_seed'])
    if min(args.episodes,args.kits,args.samples,args.problems,args.epochs,args.rounds)<1:parser.error('Budgets must be positive')
    from src.assembly.assets import bootstrap,verify
    try:
        bootstrap()
        verify()
        from src.assembly.experts import SKILLS
        if args.task=='list':
            from assembly_policy.sim.tasks import TASKS
            for arm,names in SKILLS.items():
                print('arm',arm,[(name,TASKS[name].family) for name in names])
            print('Released native neural joint-action expert; disjoint catalogs, shared frozen backbone.')
            return
        if args.task=='preflight':
            from src.assembly.profiles import qualify
            result=qualify(samples=args.samples,seed=args.seed,save=not args.verify_only)
            print('Qualified:',sum(r['success'] for r in result['trials']),'/',len(result['trials']))
            if not args.verify_only:
                from src.assembly.profiles import freeze_manifest
                print('Frozen manifest:',freeze_manifest())
            return
        if args.task=='freeze':
            from src.assembly.profiles import freeze_manifest
            print(freeze_manifest())
            return
        if args.task=='train':
            from src.assembly.head import train
            train(problems=args.problems,epochs=args.epochs,rounds=args.rounds)
            return
        from src.assembly.head import load,proxy_benchmark
        from src.assembly.profiles import load_profiles
        from src.assembly.benchmark import run
        model=load(args.checkpoint) if args.task in ('evaluate','benchmark') else None
        if args.proxy:
            if model is None:parser.error('--proxy is for evaluating a trained head')
            rows=proxy_benchmark(model,load_profiles(),range(args.seed,args.seed+args.episodes),kits=args.kits)
            print(json.dumps(rows,indent=2))
            return
        modes=('sequential','fifo','critical_path','johnson','head') if args.task=='benchmark' else (('head',) if args.task=='evaluate' else (args.scheduler,))
        if args.headless:
            _,report=run(model=model,modes=modes,seeds=range(args.seed,args.seed+args.episodes),kits=args.kits)
        else:
            from src.assembly.visualization import Display
            import mujoco.viewer
            display=Display()
            with mujoco.viewer.launch_passive(display.model,display.data) as viewer:
                viewer.cam.lookat[:]=[.45,0.,1.05]
                viewer.cam.distance,viewer.cam.azimuth,viewer.cam.elevation=2.7,0,-40
                viewer.opt.geomgroup[3:]=0
                def frame(world):
                    if not viewer.is_running():raise KeyboardInterrupt
                    with viewer.lock():display.sync(world)
                    viewer.sync();time.sleep(.04)
                _,report=run(model=model,modes=modes,seeds=range(args.seed,args.seed+args.episodes),kits=args.kits,on_frame=frame)
        if any(result['successes']<result['episodes'] for result in report['summary'].values()):
            parser.exit(1,'Some batches failed; every failure is retained in the report.\n')
    except (ValueError,RuntimeError,FileNotFoundError) as error:
        parser.exit(1,str(error)+'\n')


if __name__=='__main__':main()
