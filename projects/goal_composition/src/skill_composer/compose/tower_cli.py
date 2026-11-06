"""Commands for the separate multi-payload composition experiment."""

import argparse
from collections import deque
import json
from pathlib import Path
import statistics

from ..common import sha256, write_json
from .tower_data import DATA, PROJECT, RAW
from .tower_domain import shortest_plan, transition

RUN = PROJECT / 'outputs/tower'
COLOR_NAMES = ('red', 'blue', 'yellow')
BLOCKS = ('part0', 'part1', 'part2')
COLOR_TO_BLOCK = dict(zip(COLOR_NAMES, BLOCKS))


def main():
    from .tower_world import LAYOUTS

    parser = argparse.ArgumentParser(prog='skill-composer compose tower')
    commands = parser.add_subparsers(dest='command', required=True)
    download = commands.add_parser('download', help='Verify or fetch the two published source datasets')
    download.add_argument('--raw', type=Path, default=RAW)
    prepare = commands.add_parser('prepare', help='Prepare recorded relay and tower skill decisions')
    prepare.add_argument('--raw', type=Path, default=RAW)
    prepare.add_argument('--data', type=Path, default=DATA)
    train = commands.add_parser('train', help='Train the multi-object skill scorer')
    train.add_argument('--epochs', type=int, default=100)
    train.add_argument('--batch-size', type=int, default=128)
    for command in (train, commands.add_parser('evaluate', help='Score held-out recorded decisions')):
        command.add_argument('--data', type=Path, default=DATA)
        command.add_argument('--run', type=Path, default=RUN)

    rollout = commands.add_parser('rollout', help='Compose and execute skills in the three-block workbench')
    rollout.add_argument('--layout', choices=LAYOUTS, default='relay')
    rollout.add_argument('--order', nargs=3, choices=COLOR_NAMES, default=COLOR_NAMES,
                         metavar=('BOTTOM', 'MIDDLE', 'TOP'))
    rollout.add_argument('--controller', choices=('learned', 'guided', 'search'), default='learned')
    rollout.add_argument('--seed', type=int, default=0)
    rollout.add_argument('--viewer', action='store_true')
    rollout.add_argument('--run', type=Path, default=RUN)
    compare = commands.add_parser('compare', help='Compare greedy, guided, and shortest-plan controllers')
    compare.add_argument('--seeds', type=int, default=5)
    compare.add_argument('--seed-start', type=int, default=3000)
    compare.add_argument('--run', type=Path, default=RUN)
    args = parser.parse_args()
    try:
        _run(args, LAYOUTS)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))


def _guided(model):
    """Reuse the plan while the measured state follows its predicted effects."""
    pending = deque()
    expected = None

    def choose(state):
        nonlocal expected
        if state != expected or not pending:
            pending.clear()
            pending.extend(shortest_plan(state, rank=model.rank))
        if not pending:
            raise ValueError('The goal is already satisfied')
        action = pending.popleft()
        expected = transition(state, action)
        return action

    return choose


def _trial(rollout, controller, model, layout, seed, order, viewer=False):
    if controller == 'search':
        chooser = None
    elif controller == 'guided':
        chooser = _guided(model)
    else:
        chooser = model.choose
    result = rollout(layout, seed, chooser, viewer, order=order)
    result['controller'] = controller
    return result


def _run(args, layouts):
    if args.command == 'download':
        from .tower_data import download

        download(args.raw)
        return
    if args.command == 'prepare':
        from .tower_data import prepare

        prepare(args.raw, args.data)
        return
    if args.command in ('train', 'evaluate'):
        from .tower_policy import evaluate, train

        result = (train(args.data, args.run, args.epochs, args.batch_size) if args.command == 'train'
                  else evaluate(args.data, args.run))
        print(json.dumps(result, indent=2))
        return

    from .tower_world import rollout

    model, digest = None, None
    if args.command == 'compare' or args.controller != 'search':
        from .tower_policy import load

        model, _ = load(args.run)
        digest = sha256(args.run / 'tower.pt')
    if args.command == 'rollout':
        order = tuple(COLOR_TO_BLOCK[color] for color in args.order)
        result = _trial(rollout, args.controller, model, args.layout, args.seed, order, args.viewer)
        result['checkpoint_sha256'] = digest
        file = f'rollout_{args.controller}_{args.layout}_{"-".join(args.order)}.json'
        write_json(args.run / file, result)
        print(json.dumps({key: value for key, value in result.items() if key != 'skill_calls'}, indent=2))
        return

    if args.seeds < 1:
        raise ValueError('seeds must be positive')
    cases = [(layout, BLOCKS) for layout in layouts] + [('relay', (BLOCKS[2], BLOCKS[0], BLOCKS[1]))]
    summary, episodes = [], []
    for layout, order in cases:
        for controller in ('learned', 'guided', 'search'):
            trials = [_trial(rollout, controller, model, layout, seed, order)
                      for seed in range(args.seed_start, args.seed_start + args.seeds)]
            episodes.extend(trials)
            successes = [trial for trial in trials if trial['success']]
            row = {
                'layout': layout, 'order': order, 'controller': controller, 'episodes': len(trials),
                'successes': len(successes),
                'successes_without_sampled_arm_contact': sum(
                    trial['sampled_arm_contact_frames'] == 0 for trial in successes),
                'mean_calls_on_success': statistics.mean(len(trial['skill_calls']) for trial in successes) if successes else None,
                'mean_time_s_on_success': statistics.mean(trial['time_s'] for trial in successes) if successes else None,
            }
            summary.append(row)
            print(f'{layout} {"/".join(order)} {controller}: {len(successes)}/{len(trials)}', flush=True)
    write_json(args.run / 'comparison.json', {'checkpoint_sha256': digest, 'summary': summary, 'episodes': episodes})
