"""Run paired baseline rollouts with released pushing and lifting experts."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from src.planar.experts import FrozenExperts, REVISION, RELEASES
from src.planar.world import CollectionWorld
from src.planar.head import load_head, train


def run_episode(experts, policy, seed, cubes, max_decisions, gif=None, head=None):
    frames = []

    def capture(env):
        from PIL import Image, ImageDraw
        view = env.p.computeViewMatrix((0, 0, .85), (0, 0, 0), (0, 1, 0))
        projection = env.p.computeProjectionMatrixFOV(60, 480 / 360, .01, 3.)
        raw = env.p.getCameraImage(480, 360, viewMatrix=view, projectionMatrix=projection,
                                   renderer=env.p.ER_TINY_RENDERER)[2]
        pixels = np.asarray(raw, dtype=np.uint8).reshape(360, 480, 4)
        frame = Image.fromarray(pixels[:, :, :3])
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0, 0, 480, 35), fill='#142336')
        draw.text((10, 10), f'Released pushing + lifting | {policy} | {len(env.removed_cube_ids_set)}/{cubes} collected', fill='white')
        frames.append(frame)

    world = CollectionWorld(experts, seed, cubes, max_decisions, capture if gif else None)
    started = time.monotonic()
    try:
        if gif:
            capture(world.env)
        while not world.done:
            choice = {'push': 0, 'lift': 1, 'both': 2, 'alternate': world.decisions % 2}.get(policy)
            if policy == 'head' and head is not None:
                choice = head.choose(world.features())
            if choice is None:
                raise ValueError(f'Unknown policy: {policy}')
            world.step(choice)
        report = world.report()
        report.update(seed=seed, policy=policy, wall_seconds=time.monotonic() - started)
        experts.assert_frozen()
        if gif and frames:
            capture(world.env)
            gif.parent.mkdir(parents=True, exist_ok=True)
            frames[0].save(gif, save_all=True, append_images=frames[1:], duration=100, loop=0)
        return report
    finally:
        world.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', choices=('push', 'lift', 'both', 'alternate', 'head', 'suite'), default='both')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--train-episodes', type=int, default=64)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20000)
    parser.add_argument('--cubes', type=int, default=2)
    parser.add_argument('--max-decisions', type=int, default=60)
    parser.add_argument('--gif', type=Path)
    parser.add_argument('--output', type=Path, default=Path('outputs/planar/baselines.json'))
    args = parser.parse_args()
    if min(args.episodes, args.cubes, args.max_decisions, args.train_episodes) < 1:
        parser.error('Episode, cube, and decision counts must be positive')
    torch.set_num_threads(2)
    experts = FrozenExperts()
    head = None
    if args.train:
        train(experts, args.output.parent, args.train_episodes, args.cubes, args.max_decisions)
        return
    if args.checkpoint:
        head = load_head(args.checkpoint, experts, args.cubes, args.max_decisions)
    if args.policy == 'head' and head is None:
        parser.error('--policy head requires --checkpoint')
    policies = (('push', 'lift', 'both', 'alternate') + (('head',) if head else ())) if args.policy == 'suite' else (args.policy,)
    reports = []
    for policy in policies:
        for episode in range(args.episodes):
            gif = args.gif if episode == 0 and policy == policies[0] else None
            report = run_episode(experts, policy, args.seed + episode, args.cubes, args.max_decisions, gif, head)
            reports.append(report)
            print(json.dumps({k: v for k, v in report.items() if k != 'trace'}), flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({'upstream_revision': REVISION, 'released_experts': RELEASES,
                                              'frozen_parameter_hash': experts.initial_hash,
                                              'episodes': reports}, indent=2) + '\n')


if __name__ == '__main__':
    main()
