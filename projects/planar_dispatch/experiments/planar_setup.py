"""Fetch pinned source and two released experts, then build native pathfinding."""
import os
import subprocess
import sys
import urllib.request

from src.planar.experts import ASSETS, VENDOR, REVISION, RELEASES, digest, verify_source


def main():
    if not VENDOR.exists():
        VENDOR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', 'https://github.com/jimmyyhwu/spatial-intention-maps.git', str(VENDOR)], check=True)
        subprocess.run(['git', '-C', str(VENDOR), 'checkout', '--detach', REVISION], check=True)
    revision = subprocess.check_output(['git', '-C', str(VENDOR), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REVISION:
        raise RuntimeError(f'Expected upstream revision {REVISION}, got {revision}')
    verify_source()
    ASSETS.mkdir(parents=True, exist_ok=True)
    base = 'https://spatial-intention-maps.cs.princeton.edu/pretrained'
    for name, (run, filename, expected) in RELEASES.items():
        for url, path in (
            (f'{base}/logs/{run}/config.yml', ASSETS / f'{name}.yml'),
            (f'{base}/checkpoints/{run}/{filename}', ASSETS / f'{name}.pth.tar'),
        ):
            if not path.exists():
                temporary = path.with_suffix(path.suffix + '.part')
                urllib.request.urlretrieve(url, temporary)
                temporary.replace(path)
        if digest(ASSETS / f'{name}.pth.tar') != expected:
            raise RuntimeError(f'Invalid released weights: {name}')
    subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'],
                   cwd=VENDOR / 'shortest_paths', check=True, env=os.environ.copy())
    print('Source and released expert weights verified; pathfinding built.')


if __name__ == '__main__':
    main()
