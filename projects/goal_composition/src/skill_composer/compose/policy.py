"""Behavior cloning over executable skill calls from published annotations."""

import json
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import nn

from ..common import sha256, write_json
from .domain import FEATURES, FEATURE_NAMES, Action, State, candidates, encode


def read_examples(path):
    """Check split isolation and label legality before fitting."""
    data = json.loads(Path(path).read_text())
    rows = data['examples']
    partitions = {name: set(data['splits'][name]) for name in ('train', 'validation', 'test')}
    assigned = set().union(*partitions.values())
    if (any(not ids for ids in partitions.values())
            or sum(map(len, partitions.values())) != len(assigned)
            or assigned != {row['episode'] for row in rows}):
        raise ValueError('Episode partitions must be disjoint and cover all examples')

    observations, labels = [], []
    for row in rows:
        state = State(**row['state'])
        choices = candidates(state)
        label = Action(**row['action'])
        if label not in choices:
            raise ValueError(f'Expert call is infeasible in episode {row["episode"]}: {label}')
        observations.append(encode(state, choices))
        labels.append(choices.index(label))

    width = max(map(len, observations))
    inputs = np.zeros((len(rows), width, FEATURES), dtype=np.float32)
    legal = np.zeros((len(rows), width), dtype=bool)
    for i, observation in enumerate(observations):
        inputs[i, :len(observation)] = observation
        legal[i, :len(observation)] = True
    split_indices = {
        name: np.flatnonzero([row['episode'] in ids for row in rows])
        for name, ids in partitions.items()
    }
    return data, tuple(map(torch.from_numpy, (inputs, legal, np.array(labels, dtype=np.int64)))), split_indices


class SkillComposer(nn.Module):
    """A shared scorer for all arms and sites, with no task or arm-ID parameters."""

    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(FEATURES))

    def forward(self, features, legal):
        return (features @ self.weights).masked_fill(~legal, -torch.inf)

    @torch.no_grad()
    def choose(self, state: State) -> Action:
        choices = candidates(state)
        if not choices:
            raise ValueError('No candidate call is available')
        scores = torch.from_numpy(encode(state, choices)) @ self.weights
        return choices[int(scores.argmax())]


@torch.no_grad()
def _metrics(model, batch):
    features, legal, labels = batch
    logits = model(features, legal)
    correct = logits.argmax(-1) == labels
    branching = legal.sum(-1) > 1
    return {
        'decisions': len(labels),
        'accuracy': float(correct.float().mean()),
        'branching_decisions': int(branching.sum()),
        'branching_accuracy': float(correct[branching].float().mean()) if branching.any() else None,
        'loss': float(nn.functional.cross_entropy(logits, labels)),
    }


def train(data_path, run, epochs=100):
    if epochs < 1:
        raise ValueError('epochs must be at least 1')
    torch.set_num_threads(2)
    digest = sha256(data_path)
    _, tensors, indices = read_examples(data_path)
    if digest != sha256(data_path):
        raise ValueError('The dataset changed while it was being read')
    batches = {name: tuple(tensor[ids] for tensor in tensors) for name, ids in indices.items()}
    model = SkillComposer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    best, checkpoint = float('inf'), None
    history = []
    features, legal, labels = batches['train']
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(features, legal), labels)
        loss.backward()
        optimizer.step()
        validation = _metrics(model, batches['validation'])
        history.append({'epoch': epoch, 'train_loss': float(loss.detach()), 'validation': validation})
        if validation['loss'] < best:
            best = validation['loss']
            checkpoint = {
                'state_dict': {key: value.detach().clone() for key, value in model.state_dict().items()},
                'feature_names': list(FEATURE_NAMES),
                'dataset_sha256': digest,
                'epoch': epoch,
            }
        if epoch == 1 or epoch == epochs or epoch % 20 == 0:
            print(f'Epoch {epoch:3d}  loss {float(loss.detach()):.4f}  validation loss {validation["loss"]:.4f}', flush=True)

    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=run, suffix='.pt', delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(run / 'composer.pt')
    finally:
        temporary_path.unlink(missing_ok=True)
    write_json(run / 'training.json', {'method': 'behavior cloning', 'dataset_sha256': digest, 'history': history})
    return evaluate(data_path, run)


def load(run):
    path = Path(run) / 'composer.pt'
    if not path.is_file():
        raise ValueError(f'Missing checkpoint {path}. Run python -m skill_composer compose train first.')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['feature_names'] != list(FEATURE_NAMES):
        raise ValueError('Checkpoint does not match the current candidate features')
    model = SkillComposer()
    model.load_state_dict(checkpoint['state_dict'])
    return model.eval(), checkpoint


def evaluate(data_path, run):
    model, checkpoint = load(run)
    if sha256(data_path) != checkpoint['dataset_sha256']:
        raise ValueError('The dataset checksum does not match this checkpoint')
    data, tensors, indices = read_examples(data_path)
    report = {
        'test': _metrics(model, tuple(tensor[indices['test']] for tensor in tensors)),
        'test_episodes': len(data['splits']['test']),
        'selected_epoch': checkpoint['epoch'],
        'dataset_sha256': checkpoint['dataset_sha256'],
        'weights': dict(zip(FEATURE_NAMES, model.weights.detach().tolist())),
    }
    write_json(Path(run) / 'imitation_evaluation.json', report)
    return report
