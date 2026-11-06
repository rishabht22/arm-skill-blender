"""Load released single-robot experts without changing their learned actions."""
from contextlib import contextmanager
import hashlib
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / 'data/vendor/spatial_intention_maps'
ASSETS = ROOT / 'data/planar_research'
REVISION = '336e03a53a24192d4d7f92f06eb4c13d9f254333'
SOURCE_HASHES = {
    'envs.py': '7bfa5604470799cccc6453582f5e8184740f166db5c8e6a4065afc05a00a926f',
    'networks.py': '9fba26781beb94e362fb110ffe1fd470e58932705deeb9310f8b56602642c3b2',
    'resnet.py': '33971c35a3e58907ab06f0cc41626826bae62044c2ad05b6226bf81be73f1d4d',
    'shortest_paths/shortest_paths.pyx': '0bb80abb8cf20a5085f7396b20f9b4aacac1e0452178a485760ad683924bdce4',
}
RELEASES = {
    'pushing': ('20201214T092813073846-pushing_1-small_empty-base',
                'policy_00061500.pth.tar',
                'a12280f17498ceca7b66aea718302d967d59d6cea3649c5050a499d5b97f8f9f'),
    'lifting': ('20201217T171254022070-lifting_1-small_empty-base',
                'policy_00041000.pth.tar',
                'dec69554b8e35cb114ee25236b6588645e82f6c02aa9df7044ef1454cf350a41'),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_source():
    for filename, expected in SOURCE_HASHES.items():
        if digest(VENDOR / filename) != expected:
            raise ValueError(f'Pinned upstream source changed: {filename}')


@contextmanager
def module_alias(name, module):
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        yield
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, VENDOR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FrozenExperts:
    """Robot 0 owns pushing; robot 1 owns lifting. No cross-catalog dispatch."""
    catalog = (('pushing',), ('lifting',))

    def __init__(self):
        verify_source()
        resnet = load_module('_planar_resnet', 'resnet.py')
        with module_alias('resnet', resnet):
            network = load_module('_planar_networks', 'networks.py')
        self.networks = []
        for name, channels in (('pushing', 1), ('lifting', 2)):
            path = ASSETS / f'{name}.pth.tar'
            if digest(path) != RELEASES[name][2]:
                raise ValueError(f'Checkpoint hash mismatch: {path}')
            released = torch.load(path, map_location='cpu', weights_only=True)
            state = {k.removeprefix('module.'): v for k, v in released['state_dicts'][0].items()}
            net = network.FCN(num_input_channels=4, num_output_channels=channels)
            net.load_state_dict(state, strict=True)
            net.eval().requires_grad_(False)
            self.networks.append(net)
        self.initial_hash = self.parameter_hash()

    @torch.inference_mode()
    def action(self, robot, observation):
        if robot not in (0, 1):
            raise ValueError('Unknown robot')
        obs = np.asarray(observation, dtype=np.float32)
        if obs.shape != (96, 96, 4) or not np.isfinite(obs).all():
            raise ValueError(f'Invalid native observation: {obs.shape}')
        x = torch.from_numpy(obs.transpose(2, 0, 1).copy()).unsqueeze(0)
        # Native greedy DQN inference; the composition layer never edits pixels.
        q = self.networks[robot](x)
        return int(q.flatten().argmax())

    def parameter_hash(self):
        h = hashlib.sha256()
        for net in self.networks:
            for name, tensor in net.state_dict().items():
                h.update(name.encode())
                h.update(tensor.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def assert_frozen(self):
        assert all(not net.training for net in self.networks)
        assert all(not p.requires_grad for net in self.networks for p in net.parameters())
        assert self.parameter_hash() == self.initial_hash


def simulator_module():
    """Compatibility imports only; upstream simulation source stays on disk."""
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    source = (VENDOR / 'envs.py').read_text()
    for old, new in (
        ('import anki_vector\n', ''),
        ('import vector_utils\n', ''),
        ('from skimage.morphology.selem import disk', 'from skimage.morphology import disk'),
    ):
        if source.count(old) != 1:
            raise ValueError(f'Unexpected upstream source: {old}')
        source = source.replace(old, new)
    module = types.ModuleType('_planar_native_envs')
    module.__file__ = str(VENDOR / 'envs.py')
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    return module
