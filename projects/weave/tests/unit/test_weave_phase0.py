from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from skill_composer.data import split_episodes
from weave.eval.report import phase0_report
from weave.phase0_data import delta_actions, sha256, training_sample, verify_file


def test_audit_sample_preserves_v1_partition_and_excludes_failed_episodes() -> None:
    ids = list(range(149, -1, -1))
    metadata = [{"episode_id": i, "success": True} for i in ids]
    metadata.insert(5, {"episode_id": 999, "success": False})

    chosen, sizes = training_sample(metadata, 7, 41, 50)

    assert sizes == {"train": 105, "validation": 22, "test": 23}
    assert len({ep["episode_id"] for ep in chosen}) == 50
    assert {ep["episode_id"] for ep in chosen} <= set(
        split_episodes(ids, seed=7)["train"]
    )
    assert chosen == training_sample(metadata, 7, 41, 50)[0]


def test_delta_conversion_uses_pre_action_joints_and_preserves_gripper() -> None:
    qpos = np.zeros((3, 9), dtype=np.float32)
    qpos[1, :7] = 0.25
    actions = np.zeros((2, 8), dtype=np.float32)
    actions[0, 0] = 0.05
    actions[1, :7] = 0.25
    actions[1, 0] = 0.05
    actions[:, -1] = [-1, 1]

    commands, clipped = delta_actions(actions, qpos, 0.1)

    np.testing.assert_allclose(commands[:, 0], [0.5, -1])
    np.testing.assert_array_equal(commands[:, 1:7], 0)
    np.testing.assert_array_equal(commands[:, -1], actions[:, -1])
    assert clipped == 1


def test_manifest_validation_rejects_same_size_corruption(tmp_path: Path) -> None:
    path = tmp_path / "demo.json"
    path.write_bytes(b"abc")
    file = {"sha256": sha256(path), "bytes": 3}
    verify_file(path, file)
    path.write_bytes(b"abd")

    with pytest.raises(ValueError, match="Manifest mismatch"):
        verify_file(path, file)


def test_report_shows_missing_evidence_and_rejects_mixed_runs(tmp_path: Path) -> None:
    record = {
        "mode": "inventory",
        "git_sha": "git",
        "config_sha256": "config",
        "implementation_sha256": "source",
        "result": {"tasks": {}},
    }
    (tmp_path / "inventory.json").write_text(json.dumps(record))
    assert "Replay has no measurements." in phase0_report(tmp_path)
    assert "G0 requires a reviewed environment branch" in phase0_report(tmp_path)

    record["config_sha256"] = "other"
    (tmp_path / "different.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="different Git, config, or implementation"):
        phase0_report(tmp_path)
