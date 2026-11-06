"""Versioned controller bundles; fail closed when the environment contract changes."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "experiments/configs/mujoco_cooperation.json"
BUNDLE = ROOT / "data/skills/mujoco_cooperative/v1"
SOURCES = ("cooperative_scene.py", "cooperative.py", "composition_env.py")


def read_config():
    return json.loads(CONFIG.read_text())


def signature():
    hashes = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
              for name in SOURCES}
    versions = {name: importlib.metadata.version(name) for name in ("robosuite", "mujoco", "numpy")}
    config = read_config()
    contract = {key: config[key] for key in ("task", "skill_type", "robots", "actions",
                                             "frames_per_action", "max_decisions")}
    return dict(source_sha256=hashes, versions=versions, contract=contract)


def verify(bundle=BUNDLE):
    path = Path(bundle) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError("Frozen skills are missing. Run: python -m src.mujoco_skills.train_head preflight")
    manifest = json.loads(path.read_text())
    if manifest["signature"] != signature():
        raise RuntimeError("Frozen skill or scene contract changed. Re-run preflight and train a new head.")
    for name in SOURCES:
        content = (Path(bundle) / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != manifest["signature"]["source_sha256"][name]:
            raise RuntimeError(f"Frozen source was modified: {name}")
    return manifest


def freeze(evidence, bundle=BUNDLE):
    if not evidence or not all(row["success"] for row in evidence):
        raise ValueError("Cannot freeze unverified controllers")
    bundle = Path(bundle)
    bundle.mkdir(parents=True, exist_ok=True)
    for name in SOURCES:
        (bundle / name).write_bytes((Path(__file__).parent / name).read_bytes())
    manifest = {"format": 1, "signature": signature(), "skills": [
        {"name": "hold_container", "arm": 0, "parameters": "none; fixed controller code",
         "provenance": "project controller using robosuite Panda OSC"},
        {"name": "pick_and_place_block", "arm": 1, "parameters": "none; fixed controller code",
         "provenance": "project controller using robosuite Panda OSC"}], "evidence": evidence}
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verify(bundle)
    return manifest
