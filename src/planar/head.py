"""A small PPO dispatch head; only this module's parameters are optimized."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .world import CollectionWorld


class DispatchHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(17, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh())
        self.actor = nn.Linear(32, 3)
        self.critic = nn.Linear(32, 1)

    def forward(self, x):
        hidden = self.body(x)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)

    @torch.inference_mode()
    def choose(self, observation):
        logits, _ = self(torch.as_tensor(observation))
        return int(logits.argmax())


def load_head(path, experts, cubes, max_decisions):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['expert_hash'] != experts.initial_hash:
        raise ValueError('Head was trained with different expert parameters')
    if checkpoint['cubes'] != cubes or checkpoint['max_decisions'] != max_decisions:
        raise ValueError('Evaluation configuration differs from the trained head contract')
    head = DispatchHead()
    head.load_state_dict(checkpoint['state_dict'])
    return head.eval()


def train(experts, output, episodes=64, cubes=2, max_decisions=40, seed=0):
    torch.manual_seed(seed)
    head = DispatchHead()
    optimizer = torch.optim.Adam(head.parameters(), lr=3e-4)
    owned = {id(p) for group in optimizer.param_groups for p in group['params']}
    assert not owned.intersection(id(p) for net in experts.networks for p in net.parameters())
    reports = []
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for start in range(0, episodes, 8):
        samples = []
        for episode in range(start, min(start + 8, episodes)):
            world = CollectionWorld(experts, seed + episode, cubes, max_decisions)
            trajectory = []
            try:
                observation = world.features()
                while not world.done:
                    x = torch.as_tensor(observation)
                    with torch.no_grad():
                        logits, value = head(x)
                        distribution = Categorical(logits=logits)
                        action = distribution.sample()
                        logprob = distribution.log_prob(action)
                    observation, reward, _, info = world.step(int(action))
                    duration = max(.1, info['decision_seconds'])
                    trajectory.append((x, action, logprob, value, reward, duration))
                # Episodic objective: a decision-limit cutoff is a failure, not a
                # bootstrapped continuation. Simulated duration affects discount.
                future = 0.
                episode_samples = []
                for x, action, logprob, value, reward, duration in reversed(trajectory):
                    future = reward + (.99 ** duration) * future
                    episode_samples.append((x, action, logprob, value, future))
                samples.extend(reversed(episode_samples))
                report = world.report()
                report.update(seed=seed + episode)
                reports.append(report)
            finally:
                world.close()
        observations = torch.stack([s[0] for s in samples])
        actions = torch.stack([s[1] for s in samples])
        old_logprobs = torch.stack([s[2] for s in samples])
        values = torch.stack([s[3] for s in samples])
        returns = torch.tensor([s[4] for s in samples], dtype=torch.float32)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
        for _ in range(4):
            logits, prediction = head(observations)
            distribution = Categorical(logits=logits)
            ratio = (distribution.log_prob(actions) - old_logprobs).exp()
            surrogate = torch.minimum(ratio * advantages, ratio.clamp(.8, 1.2) * advantages)
            loss = -surrogate.mean() + .5 * (prediction - returns).square().mean() - .01 * distribution.entropy().mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), .5)
            optimizer.step()
        experts.assert_frozen()
        print(json.dumps({'training_episodes': len(reports),
                          'recent_successes': sum(r['success'] for r in reports[start:]),
                          'loss': float(loss.detach())}), flush=True)
    # Final checkpoint, selected without looking at held-out test outcomes.
    torch.save({'state_dict': head.state_dict(), 'expert_hash': experts.initial_hash,
                'cubes': cubes, 'max_decisions': max_decisions, 'seed': seed,
                'training_episodes': episodes}, output / 'head.pt')
    (output / 'training.json').write_text(json.dumps({'episodes': reports}, indent=2) + '\n')
    return head.eval()
