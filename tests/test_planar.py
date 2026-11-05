"""Released-weight integrity and actual simulator integration checks."""
import numpy as np
import pytest
import torch

from src.planar.experts import ASSETS, VENDOR, FrozenExperts
from src.planar.head import DispatchHead, load_head

native = pytest.mark.skipif(
    not (ASSETS / 'lifting.pth.tar').exists() or not (VENDOR / 'envs.py').exists(),
    reason='Run experiments.planar_setup first',
)


@native
def test_released_experts_use_disjoint_catalogs_and_stay_frozen():
    torch.set_num_threads(2)
    experts = FrozenExperts()
    assert not set(experts.catalog[0]) & set(experts.catalog[1])
    for robot in (0, 1):
        assert 0 <= experts.action(robot, np.zeros((96, 96, 4), np.float32)) < (robot + 1) * 96 * 96
    head = DispatchHead()
    optimizer = torch.optim.Adam(head.parameters())
    logits, value = head(torch.zeros(17))
    (logits.sum() + value).backward()
    optimizer.step()
    experts.assert_frozen()
    assert all(p.grad is None for net in experts.networks for p in net.parameters())


@native
def test_native_rollout_and_inactive_partner_accounting():
    from src.planar.world import CollectionWorld
    torch.set_num_threads(2)
    experts = FrozenExperts()
    for dispatch in (0, 1, 2):
        world = CollectionWorld(experts, seed=0, cubes=1, max_decisions=12)
        try:
            assert world.features().shape == (17,)
            while not world.done:
                observation, reward, _, _ = world.step(dispatch)
                assert np.isfinite(observation).all() and np.isfinite(reward)
            report = world.report()
            assert report['simulated_seconds'] > 0
            assert report['recovery_resets'] == 0
            assert report['decisions'] <= 12
            assert sum(t['seconds'] for t in report['trace']) == pytest.approx(report['simulated_seconds'])
            if dispatch == 2:
                assert report['success']
            with pytest.raises(RuntimeError):
                world.step(dispatch)
        finally:
            world.close()
    experts.assert_frozen()


def test_head_rejects_incompatible_expert_contract(tmp_path):
    class Experts:
        initial_hash = 'current'
    path = tmp_path / 'head.pt'
    torch.save({'expert_hash': 'different'}, path)
    with pytest.raises(ValueError, match='different expert'):
        load_head(path, Experts(), 2, 40)
