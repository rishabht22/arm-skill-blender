import json

import h5py
import numpy as np
import pytest

from skill_composer.common import decision_features
from skill_composer.data import annotate, prepare, split_episodes


def commands():
    # Synthetic unit-test fixture only. Training uses the published HDF5 file.
    a = np.zeros((12, 8))
    a[:, -1] = [1, 1, 1, .5, 0, -.5, -1, -1, -1, -1, -1, -1]
    a[7:10, 0] = [.1, .2, .3]
    a[10:, 0] = .3
    return a


def test_annotation_finds_actual_command_boundaries():
    labels, boundaries = annotate(commands())
    assert boundaries == (3, 7, 9)
    np.testing.assert_array_equal(labels, [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3])


def test_annotation_does_not_silently_invent_missing_grasps():
    a = commands()
    a[:, -1] = 1
    with pytest.raises(ValueError, match="manual skill annotations"):
        annotate(a)


def test_episode_splits_are_disjoint_and_reproducible():
    splits = split_episodes(range(150))
    assert splits == split_episodes(range(150))
    groups = [set(s) for s in splits.values()]
    assert [len(s) for s in groups] == [105, 22, 23]
    assert len(set.union(*groups)) == 150
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])


def test_decision_feature_layout_matches_simulation_contract():
    q = np.arange(18).reshape(2, 9)
    v = q + 100
    x = decision_features(q, v, [-1, 2], [True, False])
    assert x.shape == (48,)
    np.testing.assert_array_equal(x[:18], np.r_[q[0], v[0]])
    np.testing.assert_array_equal(x[18:36], np.r_[q[1], v[1]])
    np.testing.assert_array_equal(x[36:46].reshape(2, 5).argmax(1), [0, 3])
    np.testing.assert_array_equal(x[46:], [1, 0])


def test_preparation_aligns_current_obs_and_never_calibrates_on_heldout_episodes(tmp_path):
    splits = split_episodes(range(12))
    heldout = set(splits["test"] + splits["validation"])
    meta = {"env_info": {"env_kwargs": {"control_mode": "pd_joint_pos"}},
            "episodes": [{"episode_id": i, "success": True} for i in range(12)]}
    (tmp_path / "LiftBarrier.json").write_text(json.dumps(meta))
    with h5py.File(tmp_path / "LiftBarrier.h5", "w") as f:
        for i in range(12):
            for arm in range(2):
                g = f.require_group(f"traj_{i}")
                g.create_dataset(f"actions/panda-{arm}", data=commands())
                q = np.zeros((13, 9))
                q[:, 0] = np.arange(13) + (10000 if i in heldout else 0)
                g.create_dataset(f"obs/agent/panda-{arm}/qpos", data=q)
                g.create_dataset(f"obs/agent/panda-{arm}/qvel", data=q * 0)
    output = tmp_path / "prepared.npz"
    report = prepare(tmp_path, output)
    data = np.load(output)
    eid = splits["train"][0]
    rows = data["episode"] == eid
    # If obs[t+1] had slipped into features these would differ by one.
    np.testing.assert_array_equal(data["features"][rows, 0], data["step"][rows])
    assert report["splits"] == splits
    library = json.loads(output.with_suffix(".skills.json").read_text())
    assert library["arms"][0]["approach"][0] == 3
    assert library["arms"][0]["lift"][0] == 9
