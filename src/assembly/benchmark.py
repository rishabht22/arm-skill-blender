"""Paired physics comparisons, preserving failures and checking expert hashes."""
from datetime import datetime,timezone
import json
import time
import numpy as np
from .assets import ROOT
from .profiles import load_profiles
from .scheduling import PhysicsBatch,batch
from .planner import choose
from .service import ProductionCell


MODES=('sequential','fifo','shortest','critical_path','johnson','lookahead','head')


def run(*,model=None,modes=('fifo','head'),seeds=range(20000,20003),kits=3,on_frame=None):
    if 'head' in modes and model is None:raise ValueError('A trained head is required')
    profiles=load_profiles()
    output=ROOT/'outputs/assembly'/('benchmark_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True)
    rows=[]
    cell=ProductionCell()
    begin=time.monotonic()
    try:
        for seed in seeds:
            operations=batch(seed,kits)
            for mode in modes:
                if mode not in MODES:raise ValueError('Unknown scheduler '+mode)
                state=PhysicsBatch(operations,profiles,seed,cell=cell,on_frame=on_frame)
                inference=0.
                while not state.done:
                    start=time.perf_counter()
                    action=model.act(state) if mode=='head' else choose(state,mode)
                    inference+=time.perf_counter()-start
                    state.step(action)
                row=dict(seed=seed,scheduler=mode,backend='shared_mujoco_physics',kits=kits,
                         dispatch_compute_s=inference,jobs=list(cell.completed),**state.report())
                rows.append(row)
                (output/'episodes.json').write_text(json.dumps(rows,indent=2)+'\n')
                print(f"physics seed {seed} {mode}: success={row['success']} makespan={row['makespan_s']} failure={row['failure']}",flush=True)
                cell.world.experts.assert_frozen()
    finally:
        cell.close()
    summary={}
    for mode in modes:
        group=[r for r in rows if r['scheduler']==mode]
        good=[r for r in group if r['success']]
        summary[mode]=dict(successes=len(good),episodes=len(group),
                           mean_completion_s=float(np.mean([r['makespan_s'] for r in good])) if good else None,
                           penalized_mean_s=float(np.mean([r['makespan_s'] if r['success'] else 180. for r in group])),
                           mean_dispatch_compute_s=float(np.mean([r['dispatch_compute_s'] for r in group])))
    paired={}
    if 'head' in modes:
        for mode in modes:
            if mode=='head':continue
            baseline=[r for r in rows if r['scheduler']==mode]
            head=[r for r in rows if r['scheduler']=='head']
            pairs=[(a,b) for a,b in zip(baseline,head) if a['success'] and b['success']]
            equal_success=sum(r['success'] for r in head)>=sum(r['success'] for r in baseline)
            paired[mode]=dict(success_not_lower=equal_success,paired_successes=len(pairs),
                              mean_paired_speedup_fraction=float(np.mean([1-b['makespan_s']/a['makespan_s'] for a,b in pairs])) if pairs else None)
    report=dict(summary=summary,paired_comparisons=paired,wall_seconds=time.monotonic()-begin,
                qualification=profiles['signature'],note='Completion times are simulated seconds. Every failure is retained; speedups alone do not establish a success-preserving improvement.')
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    print('Saved',output,flush=True)
    return output,report
