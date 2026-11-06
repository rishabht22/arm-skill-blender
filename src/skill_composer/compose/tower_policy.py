"""Candidate ranking from published multi-object and relay demonstrations."""

import json
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
from torch import nn

from ..common import sha256, write_json
from .tower_domain import FEATURE_NAMES, TowerAction, TowerState, candidates, encode


class TowerComposer(nn.Module):
    def __init__(self):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(len(FEATURE_NAMES), 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1),
        )

    def forward(self, features):
        return self.scorer(features).squeeze(-1)

    @torch.no_grad()
    def rank(self, state, choices):
        return self(torch.from_numpy(encode(state, choices))).tolist()

    def choose(self, state):
        choices = candidates(state)
        if not choices:
            raise ValueError('No executable skill call remains')
        return choices[int(np.argmax(self.rank(state, choices)))]


def read_examples(path):
    dataset = json.loads(Path(path).read_text())
    examples = dataset['examples']
    splits = {name: set(dataset['splits'][name]) for name in ('train', 'validation', 'test')}
    assigned = set().union(*splits.values())
    if (any(not part for part in splits.values()) or len(assigned) != sum(map(len, splits.values()))
            or assigned != {row['episode'] for row in examples}):
        raise ValueError('Episode splits must be nonempty, disjoint, and cover the dataset')

    encoded = []
    for row in examples:
        state = TowerState(**row['state'])
        available = candidates(state)
        observed = [TowerAction(**label) for label in [row['action'], *row.get('alternatives', [])]]
        if any(action not in available for action in observed):
            raise ValueError(f'Infeasible observed call in episode {row["episode"]}, frame {row["frame"]}')
        encoded.append((encode(state, available), [call in observed for call in available]))
    width = max(len(calls) for calls, _ in encoded)
    features = np.zeros((len(examples), width, len(FEATURE_NAMES)), np.float32)
    legal = np.zeros((len(examples), width), bool)
    positive = np.zeros_like(legal)
    for index, (calls, labels) in enumerate(encoded):
        features[index, :len(calls)] = calls
        legal[index, :len(calls)] = True
        positive[index, :len(calls)] = labels
    indices = {name: np.flatnonzero([row['episode'] in ids for row in examples])
               for name, ids in splits.items()}
    return dataset, tuple(map(torch.from_numpy, (features, legal, positive))), indices


def _loss(scores, legal, positive):
    # Observed simultaneous picks can be equally valid next calls.
    legal_scores = scores.masked_fill(~legal, -torch.inf)
    expert_scores = scores.masked_fill(~positive, -torch.inf)
    return (legal_scores.logsumexp(-1) - expert_scores.logsumexp(-1)).mean()


@torch.no_grad()
def _metrics(model, tensors):
    features, legal, positive = tensors
    scores = model(features)
    chosen = scores.masked_fill(~legal, -torch.inf).argmax(-1, keepdim=True)
    correct = positive.gather(-1, chosen).squeeze(-1)
    informative = legal.sum(-1) > positive.sum(-1)
    return {
        'decisions': len(correct),
        'accuracy': float(correct.float().mean()),
        'decisions_with_unobserved_alternatives': int(informative.sum()),
        'accuracy_with_unobserved_alternatives': float(correct[informative].float().mean()) if informative.any() else None,
        'loss': float(_loss(scores, legal, positive)),
    }


def train(data_path, run, epochs=100, batch_size=128):
    if min(epochs, batch_size) < 1:
        raise ValueError('epochs and batch size must be positive')
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(19)
    digest = sha256(data_path)
    dataset, tensors, indices = read_examples(data_path)
    if digest != sha256(data_path):
        raise ValueError('Dataset changed during loading')

    model = TowerComposer()
    initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    train_ids = torch.from_numpy(indices['train'])
    validation = tuple(tensor[indices['validation']] for tensor in tensors)
    best = float('inf')
    history, steps = [], 0
    print(f'{len(train_ids)} training decisions from {len(dataset["splits"]["train"])} source episodes. '
          f'{sum(parameter.numel() for parameter in model.parameters()):,} trainable parameters.', flush=True)
    for epoch in range(1, epochs + 1):
        total_loss = 0.
        for ids in train_ids[torch.randperm(len(train_ids))].split(batch_size):
            features, legal, positive = (tensor[ids] for tensor in tensors)
            optimizer.zero_grad()
            loss = _loss(model(features), legal, positive)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(ids)
            steps += 1
        metrics = _metrics(model, validation)
        history.append({'epoch': epoch, 'train_loss': total_loss / len(train_ids), 'validation': metrics})
        if metrics['loss'] < best:
            best = metrics['loss']
            checkpoint = {
                'state_dict': {key: value.detach().clone() for key, value in model.state_dict().items()},
                'feature_names': list(FEATURE_NAMES), 'dataset_sha256': digest, 'epoch': epoch,
            }
        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(f'Epoch {epoch:3d}  train loss {total_loss / len(train_ids):.4f}  '
                  f'validation loss {metrics["loss"]:.4f}', flush=True)

    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=run, prefix='.checkpoint-') as directory:
        temporary = Path(directory) / 'tower.pt'
        torch.save(checkpoint, temporary)
        temporary.replace(run / 'tower.pt')
    write_json(run / 'training.json', {
        'history': history, 'optimizer_steps': steps, 'elapsed_seconds': time.monotonic() - started,
        'selected_checkpoint_max_weight_change': max(
            float((value - initial[key]).abs().max()) for key, value in checkpoint['state_dict'].items()),
        'dataset_sha256': digest,
    })
    return evaluate(data_path, run)


def load(run):
    path = Path(run) / 'tower.pt'
    if not path.is_file():
        raise ValueError(f'Missing checkpoint {path}. Run python -m skill_composer compose tower train first.')
    checkpoint = torch.load(path, weights_only=True, map_location='cpu')
    if checkpoint['feature_names'] != list(FEATURE_NAMES):
        raise ValueError('Checkpoint uses different scene features')
    model = TowerComposer()
    model.load_state_dict(checkpoint['state_dict'])
    return model.eval(), checkpoint


def evaluate(data_path, run):
    model, checkpoint = load(run)
    if sha256(data_path) != checkpoint['dataset_sha256']:
        raise ValueError('Checkpoint was trained on a different dataset')
    dataset, tensors, indices = read_examples(data_path)
    report = {
        'test': _metrics(model, tuple(tensor[indices['test']] for tensor in tensors)),
        'test_episodes': len(dataset['splits']['test']),
        'selected_epoch': checkpoint['epoch'],
        'dataset_sha256': checkpoint['dataset_sha256'],
    }
    write_json(Path(run) / 'imitation_evaluation.json', report)
    return report
