"""Procedural task DAGs, group isolation and immutable, hash-verified manifests."""
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import random

from .scene import CARRIERS, FAMILIAR_LAYOUTS, LAYOUTS
from .workcell import PHASES, PRIMITIVES


VERSION = 1
SPLITS = ("train", "dev", "structural", "binding", "layout", "horizon", "recovery")
PERMS = tuple(itertools.permutations(range(CARRIERS)))
TEMPLATES = ("S", "SP", "PS", "PSP", "PPS", "SPP")
FAMILIAR_BINDINGS = PERMS[:12]
NOVEL_BINDINGS = PERMS[12:]


@dataclass(frozen=True)
class Graph:
    lanes: tuple
    links: tuple


@dataclass(frozen=True)
class Instance:
    graph_id: str
    family: str
    split: str
    layout: int
    binding: tuple
    destinations: tuple
    grades: int
    recovery: tuple
    task_id: str


@dataclass(frozen=True)
class Action:
    kind: str
    job: int
    predecessors: tuple
    visit: int


def blob(obj):
    return json.dumps(obj, separators=(",", ":"), sort_keys=True).encode()


def digest(obj):
    return hashlib.sha256(blob(obj)).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(blob(value) + b"\n")
    temp.replace(path)


def transitive_reduction(lanes, links):
    vertices = [(i, j) for i, lane in enumerate(lanes) for j in range(len(lane))]
    indices = {v: i for i, v in enumerate(vertices)}
    edges = {(indices[i, j], indices[i, j + 1]) for i, j in vertices if j + 1 < len(lanes[i])}
    cross = {(indices[a, b], indices[c, d]) for a, b, c, d in links}
    edges |= cross
    reach = [0] * len(vertices)
    for a, b in edges:
        reach[a] |= 1 << b
    for k in range(len(vertices)):
        for i in range(len(vertices)):
            if reach[i] & (1 << k):
                reach[i] |= reach[k]
    if any(mask & (1 << i) for i, mask in enumerate(reach)):
        raise ValueError("cyclic graph")
    return tuple(sorted((*vertices[a], *vertices[b]) for a, b in cross
                        if not any(c != b and reach[c] & (1 << b)
                                   for x, c in edges if x == a)))


def relabel(links, perm):
    return tuple(sorted((perm[a], b, perm[c], d) for a, b, c, d in links))


def canonical(graph):
    """Colored DAG modulo job relabeling and redundant edges."""
    links = transitive_reduction(graph.lanes, graph.links)
    lanes = tuple(sorted(graph.lanes))
    return min(Graph(lanes, relabel(links, perm)).links for perm in PERMS
               if all(graph.lanes[i] == lanes[perm[i]] for i in range(CARRIERS)))


@lru_cache(maxsize=100_000)
def identity(graph):
    return digest((tuple(sorted(graph.lanes)), canonical(graph)))[:24]


@lru_cache(maxsize=100_000)
def family(graph):
    # Quotient graph on jobs. All transaction orderings and edge endpoints
    # inside jobs share this family. Canonicalize even when job names differ.
    pairs = {(a, c) for a, _, c, _ in transitive_reduction(graph.lanes, graph.links)}
    motif = min(tuple(sorted((p[a], p[c]) for a, c in pairs)) for p in PERMS)
    return digest(motif)[:16]


def assign_split(family_id):
    n = int(family_id, 16) % 100
    if n < 57:
        return "train"
    if n < 70:
        return "dev"
    if n < 82:
        return "structural"
    if n < 88:
        return "binding"
    if n < 92:
        return "layout"
    if n < 96:
        return "horizon"
    return "recovery"


def proposal(rng):
    lanes = tuple(sorted(rng.choices(TEMPLATES, k=CARRIERS)))
    order = []
    next_phase = [0] * CARRIERS
    while eligible := [i for i, j in enumerate(next_phase) if j < len(lanes[i])]:
        i = rng.choice(eligible)
        order.append((i, next_phase[i]))
        next_phase[i] += 1
    candidates = [(*a, *b) for i, a in enumerate(order) for b in order[i + 1:] if a[0] != b[0]]
    links = rng.sample(candidates, rng.randint(2, min(len(candidates), 15)))
    raw = Graph(lanes, tuple(links))
    return Graph(lanes, canonical(raw))


def longer(graph, rng):
    lanes = list(graph.lanes)
    # Extend the process history while keeping the same quotient family.
    for i in range(CARRIERS):
        lanes[i] += "P" * rng.randint(2, 3)
    if sum(lane.count("P") for lane in lanes) <= 8:
        lanes[0] += "P"
    return Graph(tuple(lanes), graph.links)


def actions(graph, recovery):
    result = []
    ends = {}
    starts = {}
    for job, lane in enumerate(graph.lanes):
        predecessors = ()
        if job in recovery:
            result.append(Action("reseat", job, (), -1))
            predecessors = (len(result) - 1,)
        visit = 0
        for phase, letter in enumerate(lane):
            starts[job, phase] = len(result)
            for kind in PHASES[letter == "P"]:
                result.append(Action(kind, job, predecessors, visit))
                predecessors = (len(result) - 1,)
            ends[job, phase] = predecessors[0]
            visit += letter == "P"
        result.append(Action("sort", job, predecessors, visit))
    for a, b, c, d in graph.links:
        idx = starts[c, d]
        current = result[idx]
        result[idx] = Action(current.kind, current.job,
                             (*current.predecessors, ends[a, b]), current.visit)
    return tuple(result)


def make_instance(gid, fid, split, layout_id, binding, destinations, grades, recovery):
    key = (gid, layout_id, binding, destinations, grades, recovery)
    return Instance(gid, fid, split, layout_id, tuple(binding), tuple(destinations), grades,
                    tuple(recovery), digest(key)[:24])


def generate(path, counts, seed, variants):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if (path / "seal.json").exists():
        raise ValueError(f"refusing to overwrite sealed manifest {path}")
    rng = random.Random(seed)
    graphs = {}
    splits = defaultdict(list)
    created = Counter()
    attempts = 0
    while any(created[s] < counts[s] for s in SPLITS):
        attempts += 1
        if attempts > 50_000_000:
            raise RuntimeError(f"unable to fill quotas: {created}")
        graph = proposal(rng)
        fid = family(graph)
        split = assign_split(fid)
        if created[split] >= counts[split]:
            continue
        if split == "horizon":
            graph = longer(graph, rng)
        gid = identity(graph)
        if gid in graphs:
            continue
        graphs[gid] = graph
        created[split] += 1
        made = set()
        while len(made) < variants:
            layout_id = rng.randrange(FAMILIAR_LAYOUTS, LAYOUTS) if split == "layout" else rng.randrange(FAMILIAR_LAYOUTS)
            pool = NOVEL_BINDINGS if split == "binding" else FAMILIAR_BINDINGS
            binding = rng.choice(pool)
            destinations = rng.choice(pool)
            recovery = tuple(sorted(rng.sample(range(CARRIERS), 2))) if split == "recovery" else ()
            if split == "train" and rng.random() < .15:
                recovery = (rng.randrange(CARRIERS),)
            item = make_instance(gid, fid, split, layout_id, binding, destinations, rng.randrange(16), recovery)
            if item.task_id not in made:
                splits[split].append(item)
                made.add(item.task_id)
    metadata = {"version": VERSION, "seed": seed, "counts": counts, "variants": variants,
                "attempts": attempts, "files": {}}
    write_json(path / "graphs.json", {key: asdict(graph) for key, graph in graphs.items()})
    for split in SPLITS:
        write_json(path / f"{split}.json", [asdict(item) for item in splits[split]])
    for name in ["graphs.json", *(f"{split}.json" for split in SPLITS)]:
        metadata["files"][name] = hashlib.sha256((path / name).read_bytes()).hexdigest()
    write_json(path / "seal.json", metadata)
    return audit(path)


def read_manifest(path, split):
    path = Path(path)
    seal = json.loads((path / "seal.json").read_bytes())
    files = {}
    for name in ("graphs.json", f"{split}.json"):
        raw = (path / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != seal["files"][name]:
            raise ValueError(f"manifest integrity failure: {name}")
        files[name] = json.loads(raw)
    graph_records = files["graphs.json"]
    instances = [Instance(**(row | {"binding": tuple(row["binding"]),
                                   "destinations": tuple(row["destinations"]), "recovery": tuple(row["recovery"])}))
                 for row in files[f"{split}.json"]]
    return {gid: Graph(tuple(graph_records[gid]["lanes"]), tuple(map(tuple, graph_records[gid]["links"])))
            for gid in {item.graph_id for item in instances}}, instances, digest(seal)


def start_seed(manifest, instance, repetition, namespace="rollout"):
    if repetition < 0:
        raise ValueError(repetition)
    return int(digest((manifest, instance.split, instance.task_id, namespace, repetition))[:16], 16)


def audit(path):
    task_ids, graph_owners, family_owners = set(), {}, {}
    ids = set()
    coverage = {}
    totals = {}
    for split in SPLITS:
        graphs, items, seal = read_manifest(path, split)
        families = set()
        primitives = set()
        for gid, graph in graphs.items():
            assert gid == identity(graph) and gid not in graph_owners
            graph_owners[gid] = split
            fid = family(graph)
            assert family_owners.setdefault(fid, split) == split == assign_split(fid)
            families.add(fid)
            assert all(lane.count("S") == 1 for lane in graph.lanes)
            if split == "horizon":
                assert sum(lane.count("P") for lane in graph.lanes) > 8
            else:
                assert all(lane in TEMPLATES for lane in graph.lanes)
        for item in items:
            graph = graphs[item.graph_id]
            assert item.family == family(graph)
            assert item == make_instance(item.graph_id, item.family, split, item.layout, item.binding, item.destinations,
                                         item.grades, item.recovery)
            assert item.task_id not in task_ids and 0 <= item.grades < 16
            task_ids.add(item.task_id)
            pool = NOVEL_BINDINGS if split == "binding" else FAMILIAR_BINDINGS
            assert item.binding in pool and item.destinations in pool
            assert (FAMILIAR_LAYOUTS <= item.layout < LAYOUTS if split == "layout"
                    else 0 <= item.layout < FAMILIAR_LAYOUTS)
            assert len(item.recovery) in ({"recovery": {2}, "train": {0, 1}}.get(split, {0}))
            seed = start_seed(seal, item, 0)
            assert seed not in ids
            ids.add(seed)
            for action in actions(graph, item.recovery):
                primitives.add(("reject" if item.grades & (1 << action.job) else "release")
                               if action.kind == "sort" else action.kind)
        coverage[split] = primitives
        totals[split] = {"graphs": len(graphs), "instances": len(items), "families": len(families),
                         "layouts": len({item.layout for item in items}), "primitives": sorted(primitives)}
    for split in SPLITS[1:]:
        assert coverage[split] <= coverage["train"] <= set(PRIMITIVES)
    return {"manifest": seal, "splits": totals}
