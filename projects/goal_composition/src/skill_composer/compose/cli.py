"""Entry point for the independent goal composition experiment."""

import argparse
import json
from pathlib import Path
import sys

from ..common import sha256, write_json
from .dataset import DATA, RAW, RUN

CASES = (('direct', 'goal'), ('relay', 'goal'), ('relay-swapped', 'goal'), ('direct', 'source'))


def _chooser(run, controller):
    if controller == 'search':
        return None
    from .policy import load

    model, _ = load(run)
    return model.choose


def main():
    if sys.argv[1:2] == ['tower']:
        sys.argv.pop(1)
        from .tower_cli import main as tower_main

        return tower_main()
    from .rollout import LAYOUTS

    parser = argparse.ArgumentParser(prog='skill-composer compose', description='Compose skill calls from a scene and a desired outcome.')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('tower', help='Three-block goal composition experiment')
    download = commands.add_parser('download', help='Download and verify published LongPipelineDelivery')
    download.add_argument('--raw', type=Path, default=RAW)
    prepare = commands.add_parser('prepare', help='Extract recorded skill events and annotate scene state')
    prepare.add_argument('--raw', type=Path, default=RAW)
    prepare.add_argument('--data', type=Path, default=DATA)
    train = commands.add_parser('train', help='Fit the goal-conditioned skill scorer')
    train.add_argument('--epochs', type=int, default=100)
    for command in (train, commands.add_parser('evaluate', help='Score a checkpoint on held-out source episodes')):
        command.add_argument('--data', type=Path, default=DATA)
        command.add_argument('--run', type=Path, default=RUN)

    rollout = commands.add_parser('rollout', help='Execute a sequence in the open workbench scene')
    rollout.add_argument('--layout', choices=LAYOUTS, default='relay')
    rollout.add_argument('--goal', choices=['goal', 'source'], default='goal')
    rollout.add_argument('--controller', choices=['learned', 'search'], default='learned')
    rollout.add_argument('--seed', type=int, default=0)
    rollout.add_argument('--viewer', action='store_true')
    rollout.add_argument('--run', type=Path, default=RUN)

    compare = commands.add_parser('compare', help='Run both controllers on the same scenes and desired outcomes')
    compare.add_argument('--seeds', type=int, default=5)
    compare.add_argument('--seed-start', type=int, default=3000)
    compare.add_argument('--run', type=Path, default=RUN)

    args = parser.parse_args()
    try:
        _run(args)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))


def _run(args):
    if args.command in ('download', 'prepare'):
        from .dataset import download, prepare

        if args.command == 'download':
            download(args.raw)
        else:
            prepare(args.raw, args.data)
        return
    if args.command in ('train', 'evaluate'):
        from .policy import evaluate, train

        result = train(args.data, args.run, args.epochs) if args.command == 'train' else evaluate(args.data, args.run)
        print(json.dumps(result, indent=2))
        return

    from .rollout import rollout

    if args.command == 'rollout':
        chooser = _chooser(args.run, args.controller)
        report = rollout(args.layout, args.seed, chooser, args.viewer, goal=args.goal)
        if chooser is not None:
            report['checkpoint_sha256'] = sha256(args.run / 'composer.pt')
        write_json(args.run / f'rollout_{args.controller}_{args.layout}_{args.goal}.json', report)
        print(json.dumps(report, indent=2))
        return

    if args.seeds < 1:
        raise ValueError('seeds must be at least 1')
    chooser = _chooser(args.run, 'learned')
    report = {'checkpoint_sha256': sha256(args.run / 'composer.pt'), 'summary': [], 'episodes': []}
    for layout, goal in CASES:
        for name, policy in (('learned', chooser), ('search', None)):
            episodes = [rollout(layout, seed, policy, goal=goal)
                        for seed in range(args.seed_start, args.seed_start + args.seeds)]
            successes = [row for row in episodes if row['success']]
            report['episodes'].extend(episodes)
            report['summary'].append({
                'layout': layout, 'goal': goal, 'controller': name,
                'runs': len(episodes), 'successes': len(successes),
                'successes_without_sampled_arm_contact': sum(
                    row['sampled_arm_contact_frames'] == 0 for row in successes),
                'mean_calls_on_success': (sum(len(row['skill_calls']) for row in successes) / len(successes)
                                          if successes else None),
            })
    write_json(args.run / 'comparison.json', report)
    print(json.dumps(report['summary'], indent=2))
