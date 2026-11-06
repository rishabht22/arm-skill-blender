"""Project published delivery and stacking recordings into relational decisions."""

from dataclasses import asdict
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request

import h5py
import numpy as np

from ..cell.dataset import _runs
from ..common import sha256, write_json
from ..data import split_episodes
from .dataset import PROJECT, RAW, REPOSITORY, REVISION, _gripper_cycle
from .dataset import download as download_delivery
from .dataset import prepare as prepare_delivery
from .tower_domain import TowerAction, TowerState, candidates, transition

DATA = PROJECT / 'data/processed/tower_decisions.json'
TASK = 'ThreeRobotsStackCube'
DELIVERY_TASK = 'LongPipelineDelivery'
UPSTREAM = f'https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{TASK}'
STACK_EPISODE_OFFSET = 10_000
ARMS = ('panda-0', 'panda-1', 'panda-2')
BLOCKS = ('cubeA', 'cubeB', 'cubeC')
CHECKSUMS = {
    f'{TASK}.json': '748babc7238ab1a6ddd67c9c2c375297b71cb1789209bac0549e8eb6226d8a8a',
    f'{TASK}.h5': '4ac615e6ed0ca257fe2be5d9182e5002b202825af1d361ec259b47ca0a2a6ef0',
}


def _verify(path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path}. Download the published tower dataset first.')
    digest = sha256(path)
    if digest != CHECKSUMS[path.name]:
        raise ValueError(f'{path} differs from pinned RoboFactory revision {REVISION}')
    return {'file': path.name, 'bytes': path.stat().st_size, 'sha256': digest}


def download(raw=RAW):
    """Fetch both releases when needed, checking each source file before use."""
    raw = Path(raw)
    download_delivery(raw)
    for filename in CHECKSUMS:
        target = raw / filename
        if not target.exists():
            print(f'Downloading {filename} from RoboFactory revision {REVISION}', flush=True)
            with tempfile.TemporaryDirectory(dir=raw, prefix='.download-') as staging:
                partial = Path(staging) / filename
                with urllib.request.urlopen(f'{UPSTREAM}/{filename}', timeout=120) as response:
                    with partial.open('wb') as stream:
                        shutil.copyfileobj(response, stream, 4 * 1024 * 1024)
                _verify(partial)
                partial.replace(target)
        else:
            _verify(target)
        print(f'Verified published {filename}', flush=True)


def _approach_and_placement(commands, preceding_release):
    close_start, close_end, release_start, release_end = _gripper_cycle(commands)
    speed = np.r_[0, np.linalg.norm(np.diff(commands[:, :7], axis=0), axis=1)]
    approach = np.flatnonzero(speed[:close_start] > .003)
    movement = _runs(speed > .003)
    lifts = [(start, end) for start, end in movement
             if close_end <= start < release_start and end - start >= 3]
    if not len(approach) or not lifts:
        raise ValueError('Missing recorded approach or post-grasp motion')
    placement = [start for start, end in movement
                 if max(lifts[0][1], preceding_release) <= start < release_start
                 and end - start >= 3]
    if not placement:
        raise ValueError('Missing recorded placement motion before release')
    return int(approach[0]), close_start, int(placement[0]), release_end


def _row(episode, source_episode, source_task, frame, source_arm, state, action, alternatives=()):
    choices = candidates(state)
    if any(choice not in choices for choice in (action, *alternatives)):
        raise ValueError(f'Recorded call conflicts with the annotated state: {action}')
    return {
        'episode': episode,
        'source_episode': source_episode,
        'source_task': source_task,
        'frame': frame,
        'source_arm': source_arm,
        'state': asdict(state),
        'action': asdict(action),
        'alternatives': [asdict(choice) for choice in alternatives],
    }


def _delivery_rows(recording):
    for example in recording['examples']:
        observed = example['state']
        state = TowerState(
            positions={'payload': observed['location']} if observed['location'] is not None else {},
            holders={'payload': observed['holder']} if observed['holder'] is not None else {},
            reach=observed['reach'],
            goals={'payload': observed['goal']},
        )
        yield _row(example['episode'], example['episode'], DELIVERY_TASK,
                   example['frame'], example['source_arm'], state,
                   TowerAction(obj='payload', **example['action']))


def _stack_rows(episode, group):
    commands = [group['actions'][arm][:] for arm in ARMS]
    lengths = {len(command) for command in commands}
    if len(lengths) != 1 or any(command.ndim != 2 or command.shape[1] != 8
                                or not np.isfinite(command).all() for command in commands):
        raise ValueError('Expected synchronized (T, 8) joint and gripper actions')

    picks, closes, placements = [], [], []
    preceding_release = 0
    for command in commands:
        pick, close, place, preceding_release = _approach_and_placement(command, preceding_release)
        picks.append(pick)
        closes.append(close)
        placements.append(place)
    if len(set(closes)) != 1 or max(picks) >= min(closes):
        raise ValueError('Expected the recorded parallel pick group and synchronized gripper close')

    source_pads = tuple(f'source{arm}' for arm in range(len(ARMS)))
    goals = dict(zip(BLOCKS, ('goal', *BLOCKS[:-1])))
    reach = {site: (arm,) for arm, site in enumerate(source_pads)}
    reach['goal'] = tuple(range(len(ARMS)))
    state = TowerState(positions=dict(zip(BLOCKS, source_pads)), holders={}, reach=reach, goals=goals)
    key = STACK_EPISODE_OFFSET + episode

    # The published solver starts these picks together. Linearize their
    # commuting effects and preserve the other observed picks as valid labels.
    order = sorted(range(len(ARMS)), key=lambda arm: (picks[arm], arm))
    pick_actions = [TowerAction('pick', arm, BLOCKS[arm], source_pads[arm]) for arm in order]
    rows = []
    for index, action in enumerate(pick_actions):
        rows.append(_row(key, episode, TASK, picks[action.arm], ARMS[action.arm], state, action,
                         alternatives=pick_actions[index + 1:]))
        state = transition(state, action)

    for arm, obj in enumerate(BLOCKS):
        target = goals[obj]
        action = TowerAction('place' if target == 'goal' else 'stack', arm, obj, target)
        rows.append(_row(key, episode, TASK, placements[arm], ARMS[arm], state, action))
        state = transition(state, action)
    if not all(state.positions.get(obj) == target for obj, target in goals.items()):
        raise ValueError('Annotated placements did not complete the recorded stack')
    return rows, next(iter(lengths))


def prepare(raw=RAW, output=DATA):
    raw, output = Path(raw), Path(output)
    hdf5_path, metadata_path = raw / f'{TASK}.h5', raw / f'{TASK}.json'
    source_files = [_verify(path) for path in (hdf5_path, metadata_path)]
    metadata = json.loads(metadata_path.read_text())
    if metadata['env_info']['env_kwargs']['control_mode'] != 'pd_joint_pos':
        raise ValueError('Stack annotation requires the published joint-position controls')
    episode_ids = sorted(int(row['episode_id']) for row in metadata['episodes'] if row.get('success') is True)
    stack_splits = split_episodes(episode_ids)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='.tower-prepare-') as staging:
        # Rebuild the delivery projection from verified raw files without
        # replacing the earlier workbench dataset or its checkpoint.
        delivery = prepare_delivery(raw, Path(staging) / 'delivery.json')
        rows = list(_delivery_rows(delivery))
        frames = 0
        with h5py.File(hdf5_path, 'r') as recordings:
            for episode in episode_ids:
                try:
                    decisions, length = _stack_rows(episode, recordings[f'traj_{episode}'])
                except (KeyError, ValueError) as error:
                    raise ValueError(f'Cannot annotate {TASK} episode {episode}: {error}') from error
                rows.extend(decisions)
                frames += length

        splits = {
            split: sorted(delivery['splits'][split] + [STACK_EPISODE_OFFSET + eid for eid in stack_splits[split]])
            for split in stack_splits
        }
        report = {
            'source': REPOSITORY,
            'revision': REVISION,
            'sources': [
                {key: delivery[key] for key in ('task', 'source_url', 'source_files', 'source_episodes', 'source_frames')},
                {'task': TASK, 'source_url': UPSTREAM, 'source_files': source_files,
                 'source_episodes': len(episode_ids), 'source_frames': frames},
            ],
            'source_episodes': delivery['source_episodes'] + len(episode_ids),
            'source_frames': delivery['source_frames'] + frames,
            'decision_events': len(rows),
            'distinct_abstract_observations': len({json.dumps(row['state'], sort_keys=True) for row in rows}),
            'branching_decision_events': sum(len(candidates(TowerState(**row['state']))) > 1 for row in rows),
            'alternative_label_events': sum(bool(row['alternatives']) for row in rows),
            'annotation': (
                'The two delivery stages retain the existing verified annotation. Stacking labels project '
                'the original parallel pick group and ordered releases into relational calls. Concurrent '
                'picks are linearized by observed approach and then arm ID, with other observed pending '
                'picks marked as valid labels. Goals, support bindings, and site reachability come from '
                'task annotations. Frames mark motion inferred from recorded commands. Episode splits '
                'keep all labels from each original recording together.'
            ),
            'splits': splits,
            'examples': rows,
        }
        temporary = Path(staging) / output.name
        write_json(temporary, report)
        temporary.replace(output)

    print(f'Prepared {report["source_episodes"]} published episodes and {len(rows)} relational decisions at {output}',
          flush=True)
    return report
