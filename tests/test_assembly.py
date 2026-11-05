"""Scheduling invariants, frozen experts, shared contact physics and deployment."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch

from src.assembly.experts import SKILLS,TASKS
from src.assembly.scheduling import batch,PlanningModel
from src.assembly.planner import choose,lookahead
from src.assembly.head import DispatchHead,inputs
from src.assembly.assets import VENDOR,PLUGIN


def fake_profiles():
    return {'skills':{name:{'mean_seconds':3.+i%4,'std_seconds':.2} for i,name in enumerate(TASKS)}}


def test_exclusive_real_assembly_catalogs_and_reentrant_workorders():
    assert len(SKILLS[0])==len(SKILLS[1])==6
    assert not set(SKILLS[0])&set(SKILLS[1])
    ops=batch(7,3)
    assert len(ops)==12
    for kit in range(3):
        route=ops[4*kit:4*kit+4]
        assert [j.arm for j in route]==[0,1,0,1]
        assert [j.predecessor for j in route]==[-1,4*kit,4*kit+1,4*kit+2]
        assert all(j.skill in SKILLS[j.arm] for j in route)


def test_scheduling_masks_resources_precedence_and_release():
    s=PlanningModel(batch(2,3),fake_profiles(),stochastic=False)
    with pytest.raises(ValueError):s.step(1)
    while not s.done:
        ready=s.ready()
        for i in ready:
            j=s.operations[i]
            assert s.active[j.arm]<0
            assert j.predecessor<0 or s.status[j.predecessor]==2
        s.step(choose(s,'critical_path'))
    assert s.success and (s.status==2).all()
    with pytest.raises(RuntimeError):s.step(0)


def test_planner_has_no_access_to_hidden_future_durations():
    a=PlanningModel(batch(7,3),fake_profiles(),seed=1)
    b=PlanningModel(batch(7,3),fake_profiles(),seed=99)
    assert not np.array_equal(a.durations,b.durations)
    np.testing.assert_array_equal(inputs(a),inputs(b))
    assert lookahead(a,width=4)==lookahead(b,width=4)


def test_parallel_planning_and_legal_head_actions():
    p=fake_profiles();ops=batch(0,2);times={}
    model=DispatchHead().eval()
    for mode in ('sequential','critical_path','head'):
        s=PlanningModel(ops,p,stochastic=False)
        while not s.done:
            action=model.act(s) if mode=='head' else choose(s,mode)
            assert s.mask()[action]
            s.step(action)
        times[mode]=s.time
    assert times['critical_path']<times['sequential']


def test_dispatch_head_is_permutation_equivariant():
    torch.manual_seed(3)
    model=DispatchHead().eval()
    x=torch.randn(1,9,29);present=torch.ones(1,9,dtype=torch.bool)
    order=torch.randperm(9)
    a=model(x,present);b=model(x[:,order],present)
    torch.testing.assert_close(a[:,:9][:,order],b[:,:9],rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(a[:,-1],b[:,-1],rtol=1e-5,atol=1e-6)


native=pytest.mark.skipif(not (VENDOR/'UPSTREAM.json').is_file() or not PLUGIN.is_file(),reason='Run precision-assembly setup first')


@native
def test_upstream_assets_and_qualification_keep_all_failures():
    from src.assembly.assets import verify
    from src.assembly.profiles import load_profiles
    manifest=verify();p=load_profiles()
    assert manifest['commit']=='eb5d15e004591d380c0d521da67e025ebee37703'
    assert len(p['trials'])==sum(v['trials'] for v in p['skills'].values())
    assert all(v['success_rate']>=.8 for v in p['skills'].values())
    assert len(p['skills'])==12


@native
def test_shared_world_preserves_native_joint_limits_and_control():
    import mujoco
    from src.assembly.world import JointWorld
    w=JointWorld()
    try:
        m=w._model(('R4','Gs'))
        for arm in (0,1):
            native=w.shadow.models[0].model
            ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'a{arm}_joint{i}') for i in range(1,8)]
            np.testing.assert_allclose(m.jnt_range[ids],native.jnt_range[:7])
            np.testing.assert_allclose(m.actuator_gainprm[8*arm:8*arm+8],native.actuator_gainprm)
            np.testing.assert_allclose(m.actuator_biasprm[8*arm:8*arm+8],native.actuator_biasprm)
        assert m.nu==16 and m.nq==32
        assert not any((mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or '').startswith('wall') for i in range(m.ngeom))
        w.experts.assert_frozen()
    finally:w.close()


@native
def test_workpiece_exchange_does_not_teleport_robots_or_other_part():
    from src.assembly.service import ProductionCell,Job,loose_state
    c=ProductionCell()
    try:
        jobs=[Job(0,0,'R4',loose_state('R4',30)),Job(1,1,'Gs',loose_state('Gs',40))]
        c.reset(jobs);c.start(jobs[0]);c.step()
        e=c.world.shadow
        before=e.state.copy();history=e.hist[0].copy();clock=c.world.data.time
        c.start(Job(2,1,'N8',loose_state('N8',41)))
        np.testing.assert_allclose(e.state[:,e.sq],before[:,e.sq],atol=1e-12)
        np.testing.assert_allclose(e.state[0,e.sp],before[0,e.sp],atol=1e-12)
        np.testing.assert_allclose(e.hist[0],history,atol=1e-12)
        assert c.world.data.time==clock
    finally:c.close()


@native
def test_two_frozen_experts_assemble_and_release_in_one_physics_world():
    from src.assembly.service import ProductionCell,Job,loose_state
    c=ProductionCell()
    try:
        jobs=[Job(0,0,'R12',loose_state('R12',31)),Job(1,1,'Gm',loose_state('Gm',41))]
        c.reset(jobs)
        for j in jobs:c.start(j)
        for _ in range(1000):
            c.step()
            if c.failed or len(c.completed)==2:break
        assert c.failed is None,c.failed
        assert len(c.completed)==2
        assert all(j['error_m']<.002 for j in c.completed)
        c.world.experts.assert_frozen()
    finally:c.close()


def test_launcher_defaults_to_new_experiment_and_mac_viewer(tmp_path,monkeypatch):
    from experiments import run
    monkeypatch.setattr(run,'ROOT',tmp_path)
    folder=tmp_path/'.venv-mujoco/bin';folder.mkdir(parents=True)
    for name in ('python','mjpython'):(folder/name).touch()
    train=run.command_for('train',[],platform='darwin')
    assert train[-2:]==['experiments.assembly','train']
    assert train[0].endswith('/python')
    assert run.command_for('evaluate',[],platform='darwin')[0].endswith('/mjpython')
    assert run.command_for('evaluate',['--headless'],platform='darwin')[0].endswith('/python')
    assert run.command_for('evaluate',['--help'],platform='darwin')[0].endswith('/python')
    assert run.command_for('evaluate',[],platform='linux')[0].endswith('/python')


@native
def test_saved_head_is_frozen_and_valid_for_variable_batch_sizes():
    from src.assembly.head import load
    from src.assembly.profiles import load_profiles
    model=load();profiles=load_profiles()
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    for kits in (2,3,6):
        state=PlanningModel(batch(15,kits),profiles,seed=15)
        action=model.act(state)
        assert state.mask()[action]
