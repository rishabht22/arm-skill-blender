"""Fetch the pinned released assembly expert and build native contacts on Mac/Linux."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

from src.assembly.assets import VENDOR, verify, build_plugin

REPOSITORY = 'https://github.com/suniique/assembly-nist'
REVISION = 'eb5d15e004591d380c0d521da67e025ebee37703'


def download():
    url = 'https://api.github.com/repos/suniique/assembly-nist/git/trees/' + REVISION + '?recursive=1'
    tree = json.load(urllib.request.urlopen(url, timeout=40))
    files = [f for f in tree['tree'] if f['type'] == 'blob' and
             (f['path'].startswith(('assembly_policy/', 'assets/', 'plugin/'))
              or f['path'] in ('LICENSE', 'THIRD_PARTY.md', 'README.md', 'CITATION.bib', 'checkpoints/e8192_it12500.pt'))]
    VENDOR.mkdir(parents=True, exist_ok=True)
    def fetch(entry):
        path = VENDOR / entry['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file():
            content = path.read_bytes()
        else:
            target = f'https://raw.githubusercontent.com/suniique/assembly-nist/{REVISION}/' + entry['path']
            content = urllib.request.urlopen(target, timeout=60).read()
        git_hash = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
        if git_hash != entry['sha']:
            raise RuntimeError('Upstream asset checksum mismatch: ' + entry['path'])
        if not path.is_file():
            path.write_bytes(content)
        return dict(path=entry['path'], sha256=hashlib.sha256(content).hexdigest(), size=len(content))
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(fetch, files))
    (VENDOR / 'UPSTREAM.json').write_text(json.dumps(dict(repository=REPOSITORY, commit=REVISION, files=rows), indent=2) + '\n')
    print(f'Verified {len(rows)} pinned files ({sum(r["size"] for r in rows) / 1e6:.1f} MB).')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-only', action='store_true')
    args = parser.parse_args()
    if not args.build_only:
        download()
    verify()
    build_plugin()


if __name__ == '__main__':
    main()
