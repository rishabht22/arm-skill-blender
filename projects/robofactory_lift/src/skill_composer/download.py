"""Fetch the actual public demonstrations, with a pinned upstream revision."""
import urllib.request
from pathlib import Path

from .common import RAW, sha256, write_json

REPO = "FACEONG/RoboFactory_Dataset"
REVISION = "63f8a85"  # immutable upstream git revision, not the moving main branch
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/LiftBarrier"
EXPECTED = {
    "LiftBarrier.json": "91a44d5c59667337f74565f7897cd60a11aab35644865ca05ad52939ad24ec15",
    "LiftBarrier.h5": "fcc09ee8f0f4bcbfe46e41c467121d8ef565f4f2ff8699526d8a4a2e8d672226",
}


def download(destination=RAW):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for name in ("LiftBarrier.json", "LiftBarrier.h5"):
        target = destination / name
        digest = sha256(target) if target.exists() else None
        if digest != EXPECTED[name]:
            partial = target.with_suffix(target.suffix + ".part")
            print(f"Downloading {name} from {REPO}@{REVISION}", flush=True)
            request = urllib.request.Request(f"{BASE}/{name}", headers={"User-Agent": "arm-skill-composer/0.1"})
            with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as stream:
                expected = response.headers.get("Content-Length")
                total = 0
                while block := response.read(1024 * 1024):
                    stream.write(block)
                    total += len(block)
            if expected is not None and total != int(expected):
                raise ValueError(f"Incomplete download: {total} / {expected} bytes")
            digest = sha256(partial)
            if digest != EXPECTED[name]:
                raise ValueError(f"Downloaded {name} does not match the pinned public dataset SHA256")
            partial.replace(target)
        records.append({"name": name, "bytes": target.stat().st_size, "sha256": digest})
    write_json(destination / "manifest.json", {"repository": REPO, "revision": REVISION, "license": "MIT", "files": records})
    print(f"Published demonstrations ready in {destination}")
