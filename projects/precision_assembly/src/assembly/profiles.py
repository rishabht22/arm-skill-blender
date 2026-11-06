"""Unfiltered physical qualification and measured skill-duration profiles."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from .assets import ROOT, verify
from .experts import SKILLS, TASKS, CHECKPOINT
from .service import ProductionCell, Job, loose_state

PROFILE = ROOT / 'data/assembly/profiles.json'


def signature():
    folder = Path(__file__).parent
    return dict(checkpoint_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
                catalogs={str(a): list(s) for a, s in SKILLS.items()},
                adapter_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in folder.glob('*.py') if p.name in ('assets.py','experts.py','world.py','service.py')})


def qualify(samples=5, seed=1000, save=True):
    if samples < 3:
        raise ValueError('At least three unfiltered trials per skill are required')
    verify()
    start = time.monotonic()
    cell = ProductionCell()
    rows = []
    base = {a: loose_state(SKILLS[a][0], seed + 300 + a) for a in (0, 1)}
    try:
        for arm, skills in SKILLS.items():
            for name in skills:
                for trial in range(samples):
                    value = seed + TASKS.index(name) * 100 + trial
                    state = loose_state(name, value)
                    jobs = [Job(i, i, SKILLS[i][0], base[i]) for i in (0, 1)]
                    jobs[arm] = Job(arm, arm, name, state)
                    cell.reset(jobs)
                    cell.start(jobs[arm])
                    for _ in range(1000):
                        if cell.failed or cell.completed:
                            break
                        cell.step()
                    result = dict(skill=name, arm=arm, seed=value, success=bool(cell.completed) and cell.failed is None,
                                  seconds=cell.time, failure=cell.failed,
                                  final_q=cell.world.shadow.state[arm, cell.world.shadow.sq].tolist())
                    rows.append(result)
                    print(f"qualify {name} {trial+1}/{samples}: {result['success']} {result['seconds']:.2f}s {result['failure']}", flush=True)
    finally:
        cell.world.experts.assert_frozen()
        cell.close()
    skills = {}
    for name in TASKS:
        r = [x for x in rows if x['skill'] == name]
        good = [x for x in r if x['success']]
        skills[name] = dict(success_rate=len(good)/len(r), trials=len(r),
                            mean_seconds=float(np.mean([x['seconds'] for x in good])) if good else None,
                            std_seconds=float(np.std([x['seconds'] for x in good])) if good else None,
                            final_q=np.mean([x['final_q'] for x in good],axis=0).tolist() if good else None)
    passed = all(s['success_rate'] >= .8 for s in skills.values())
    manifest = dict(signature=signature(), passed=passed, trials=rows, skills=skills,
                    seconds=time.monotonic()-start, reset_family='fresh_loose_parts_home_robot',
                    qualification='neural assembly plus timed entry/exit; released assembly error <= 2mm')
    PROFILE.parent.mkdir(parents=True,exist_ok=True)
    destination=PROFILE if save else PROFILE.with_name(f'verification_{int(time.time())}.json')
    destination.write_text(json.dumps(manifest,indent=2)+'\n')
    if not passed:
        raise RuntimeError('Some assembly skills failed qualification. Profiles record every trial; do not train a head yet.')
    return manifest


def load_profiles():
    if not PROFILE.is_file():
        raise FileNotFoundError('Run: python -m experiments.assembly preflight')
    profile = json.loads(PROFILE.read_text())
    if not profile['passed'] or profile['signature'] != signature():
        raise RuntimeError('Native assembly qualification is missing or stale; run preflight again')
    return profile


def freeze_manifest():
    """Record a qualified immutable release; no base weights are optimized."""
    profile=load_profiles()
    data=dict(signature=profile['signature'],status='frozen_released_expert',
              upstream_revision=json.loads((ROOT/'data/vendor/assembly_nist/UPSTREAM.json').read_text())['commit'],
              pretrained_backbone='one released multi-task neural expert; disjoint role catalogs',
              qualification_successes=sum(r['success'] for r in profile['trials']),
              qualification_trials=len(profile['trials']))
    path=PROFILE.with_name('frozen.json')
    path.write_text(json.dumps(data,indent=2)+'\n')
    return path
