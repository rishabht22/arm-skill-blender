"""Fetch inference metadata from the official ZIP without downloading episodes."""

import io
from pathlib import Path
import re
import subprocess
import zipfile


class RangeReader(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        headers = subprocess.check_output(
            ["curl", "-fsSLI", "--max-time", "60", url], text=True
        )
        sizes = re.findall(r"(?im)^content-length:\s*(\d+)", headers)
        if not sizes:
            raise RuntimeError("Archive host did not provide Content-Length")
        self.size = int(sizes[-1])
        self.position = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if position < 0:
            raise ValueError("negative seek")
        self.position = position
        return position

    def read(self, size=-1):
        count = max(0, min(self.size - self.position, size if size >= 0 else self.size))
        if not count:
            return b""
        if count > 256 * 1024 * 1024:
            raise ValueError("Refusing to fetch more than 256 MiB in one metadata request")
        data = subprocess.check_output([
            "curl", "-fsSL", "--retry", "3", "--max-time", "120",
            "--max-filesize", str(count), "--range",
            f"{self.position}-{self.position + count - 1}", self.url,
        ])
        if len(data) != count:
            raise RuntimeError("Archive host did not honor the requested byte range")
        self.position += count
        return data


def fetch_metadata(asset_root):
    root = Path(asset_root).expanduser().resolve() / "datasets/task_D_D"
    wanted = [
        "validation/.hydra/merged_config.yaml",
        "validation/statistics.yaml",
        "training/statistics.yaml",
    ]
    if all((root / name).is_file() for name in wanted):
        print(f"Inference metadata already exists: {root}")
        return
    url = "http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip"
    with RangeReader(url) as stream, zipfile.ZipFile(stream) as archive:
        for name in wanted:
            matches = [entry for entry in archive.infolist() if entry.filename.endswith("/" + name)]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one metadata file for {name}, found {len(matches)}")
            entry = matches[0]
            if entry.file_size > 10 * 1024 * 1024:
                raise ValueError(f"Unexpectedly large metadata file: {entry.filename}")
            destination = root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(entry))
            print(f"Saved {destination}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset_root", type=Path)
    fetch_metadata(parser.parse_args().asset_root)
