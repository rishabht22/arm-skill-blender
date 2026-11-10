"""Regression tests for mechanical skills, replay and split isolation."""
import json

import mujoco as mj
import numpy as np
import pytest

from larger_scale.checks import controls
from larger_scale.protocol import (
    Graph, SPLITS, audit, family, generate, identity, make_instance, relabel,
)
from larger_scale.rollout import replay, run, save_recording
from larger_scale.workcell import PHASES, PRIMITIVES, Workcell


def test_canonical_graph_identity_ignores_job_labels_and_redundant_edges():
    lanes = ("SP", "PS", "SP", "S")
    edges = ((0, 1, 1, 0), (1, 1, 2, 0))
    graph = Graph(lanes, edges)
    perm = (2, 1, 3, 0)
    permuted = tuple(lanes[perm.index(i)] for i in range(4))
    augmented = (*edges, (0, 1, 2, 0))
    equivalent = Graph(permuted, relabel(augmented, perm))
    assert identity(graph) == identity(equivalent)
    assert family(graph) == family(equivalent)
    with pytest.raises(ValueError, match="cyclic"):
        identity(Graph(lanes, (*edges, (2, 0, 0, 0))))


def test_sealed_manifest_hash_and_split_isolation(tmp_path):
    counts = dict(zip(SPLITS, (80, 12, 12, 12, 12, 12, 12)))
    summary = generate(tmp_path, counts, 748731, 2)
    assert sum(item["graphs"] for item in summary["splits"].values()) == sum(counts.values())
    assert summary["splits"]["train"]["primitives"] == sorted(PRIMITIVES)
    path = tmp_path / "structural.json"
    rows = json.loads(path.read_bytes())
    rows[0]["grades"] ^= 1
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="manifest integrity"):
        audit(tmp_path)


def physical_clone(source):
    """A fresh simulator starts exactly at one primitive's input boundary."""
    clone = Workcell(0)
    clone.reset(1801, source.bindings, 5, (0,))
    spec = mj.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mj.mj_stateSize(source.model, spec))
    mj.mj_getState(source.model, source.data, state, spec)
    clone.model.eq_data[:] = source.model.eq_data
    mj.mj_setState(clone.model, clone.data, state, spec)
    mj.mj_forward(clone.model, clone.data)
    clone.reads = source.reads.copy()
    clone.cycles = source.cycles.copy()
    clone.index_count = source.index_count
    clone._rotor_command = source._rotor_command
    return clone


def test_all_primitives_from_isolated_physical_boundaries():
    source = Workcell(0)
    source.reset(1801, (0, 1, 2, 3), 5, (0,))
    verified = set()
    for job, sequence in ((0, ("reseat", *PHASES[0], *PHASES[1], "reject")),
                          (1, (*PHASES[1], *PHASES[0], "release"))):
        for kind in sequence:
            clone = physical_clone(source)
            outcome = clone.execute(kind, job)
            assert outcome.ok, (job, kind, outcome)
            assert source.execute(kind, job).ok
            verified.add(kind)
    assert verified == set(PRIMITIVES)


def test_mechanical_interlock_counterexamples():
    result = controls(Workcell(0))
    assert len(result) == 8
    assert all(row["passed"] for row in result), result


def test_matched_start_invalid_dependency_and_recording(tmp_path):
    graph = Graph(("SP", "S", "S", "S"), ((1, 0, 0, 0),))
    item = make_instance(identity(graph), family(graph), "dev", 0, (0, 1, 2, 3),
                         (1, 0, 3, 2), 5, ())
    expert, trace = run(graph, item, 0, "regression", "expert", record=True)
    fixed, _ = run(graph, item, 0, "regression", "fixed")
    symbolic, _ = run(graph, item, 0, "regression", "symbolic")
    assert expert["success"] and symbolic["success"]
    assert fixed["violations"] == 1 and fixed["completed"] == 0
    assert expert["start"] == fixed["start"] == symbolic["start"]
    warm = Workcell(item.layout)
    run(graph, item, 0, "regression", "symbolic", cell=warm)
    repeated, _ = run(graph, item, 0, "regression", "expert", cell=warm)
    assert repeated["final_physics"] == expert["final_physics"]
    path = tmp_path / "expert.npz"
    save_recording(path, expert, trace)
    assert replay(path, item, "regression")["final_physics"] == expert["final_physics"]
