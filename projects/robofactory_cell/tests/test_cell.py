import json
import numpy as np
import torch

from skill_composer.cell.program import Node,source_program,composed_program,features,MAX_NODES
from skill_composer.cell.policy import GraphPolicy
from skill_composer.cell.world import AssemblyCell
from skill_composer.cell.dataset import TASKS


def test_new_compositions_have_more_operations_than_every_source_graph():
    source=['PickMeat','StackCube','PassShoe','PlaceFood','TwoRobotsStackCube']
    assert max(len(source_program(t)) for t in source)==4
    assert len(composed_program('relay_container',4))==17
    assert len(composed_program('relay_stack',3))==12
    for name in ['relay_container','relay_stack']:
        nodes=composed_program(name,3)
        for i,node in enumerate(nodes):assert all(0<=d<i for d in node.deps)
    assert not {'relay_stack','relay_container'} & set(TASKS)


def test_dependency_readiness_is_observed_not_used_as_a_policy_mask():
    nodes=source_program('PassShoe')
    x,m=features(nodes,[0,0,0,0])
    assert x[1,2,9]==0  # Receiver's pick depends on supplier placing first.
    assert m[1,2]       # The policy must learn not to request it prematurely.
    assert m[1,MAX_NODES]


def test_graph_actor_is_equivariant_to_candidate_permutation():
    torch.manual_seed(4)
    model=GraphPolicy().eval()
    x,m=features(composed_program('relay_stack',3),[0]*12)
    x=torch.from_numpy(x)[None];m=torch.from_numpy(m)[None]
    p=torch.randperm(MAX_NODES)
    with torch.no_grad():
        original=model(x,m)
        changed=model(x[:,:,p],torch.cat([m[:,:,:MAX_NODES][:,:,p],m[:,:,-1:]],-1))
    torch.testing.assert_close(changed[:,:,:MAX_NODES],original[:,:,:MAX_NODES][:,:,p])
    torch.testing.assert_close(changed[:,:,-1],original[:,:,-1])


def test_low_level_skill_rejects_unmet_cross_arm_preconditions():
    world=AssemblyCell(source_program('PassShoe'))
    world.dispatch([-1,2])
    assert world.rejected==1
    assert world.active==[None,None]
    assert np.all(world.status==0)


def test_frozen_pick_and_place_actually_move_the_object():
    nodes=[Node('pick',0),Node('place',0,deps=(0,))]
    world=AssemblyCell(nodes)
    initial=world.data.body('part0').xpos.copy()
    for _ in range(250):
        world.dispatch([next((i for i in range(2) if world.can_start(i)),-1),-1])
        world.step()
        if all(world.status==2):break
    for _ in range(10):world.step()
    report=world.result()
    assert report['success'],report
    assert np.linalg.norm(world.data.body('part0').xpos-initial)>.15
    assert report['placement_errors_m']['part0']<.01
    assert not report['sampled_arm_contact_frames']
    assert json.dumps(report)


def test_arm_delay_is_an_executor_availability_mask_not_a_goal_guard():
    world=AssemblyCell(source_program('TwoRobotsStackCube'),delay=2.)
    _,m=world.observation()
    assert not m[1,:-1].any()
    assert m[1,-1]
