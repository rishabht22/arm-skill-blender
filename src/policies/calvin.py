"""Load the frozen HULC controller with its released observation transforms."""

from pathlib import Path
from pydoc import locate
from types import SimpleNamespace
from src.runtime import resolve_device, resolve_renderer


def load_hulc(asset_root, *, device="auto", renderer="auto", show_gui=False):
    root = Path(asset_root).expanduser().resolve()
    train_folder = root / "models/HULC_D_D"
    dataset = root / "datasets/task_D_D"
    config = train_folder / ".hydra/config.yaml"
    required = (config, dataset / "validation/.hydra/merged_config.yaml",
                dataset / "validation/statistics.yaml", dataset / "training/statistics.yaml")
    if not all(path.is_file() for path in required):
        raise FileNotFoundError(
            f"Missing HULC/CALVIN assets under {root}. Run experiments/setup_calvin.sh "
            "with sources, install, model, metadata, and language, in that order."
        )
    import torch
    import hydra
    import calvin_agent
    import hulc  # Registers the checkpoint's model package for the native loader.
    from torchvision.transforms import Compose
    from omegaconf import OmegaConf
    from calvin_agent.datasets.utils.episode_utils import load_dataset_statistics
    from src.envs.calvin_portable import PortableCalvinEnv
    from calvin_agent.utils.utils import get_last_checkpoint

    device = resolve_device(device)
    renderer = resolve_renderer(renderer, device=device)
    print(f"Policy device: {device}; CALVIN renderer: {renderer}", flush=True)
    checkpoint = get_last_checkpoint(train_folder)
    if checkpoint is None:
        raise FileNotFoundError(f"No released checkpoint found in {train_folder}")
    cfg = OmegaConf.load(config)
    if list(cfg.datamodule.observation_space.actions) != ["rel_actions"] or cfg.datamodule.action_space != 7:
        raise ValueError("The released CALVIN controller must use the original seven relative actions")
    transforms = load_dataset_statistics(dataset / "training", dataset / "validation", cfg.datamodule.transforms)
    val_transforms = {
        key: Compose([hydra.utils.instantiate(item) for item in items])
        for key, items in transforms.val.items()
    }
    # The native wrapper only needs these four attributes, not replay episodes.
    inputs = SimpleNamespace(
        abs_datasets_dir=dataset / "validation",
        observation_space=cfg.datamodule.observation_space,
        proprio_state=cfg.datamodule.proprioception_dims,
        transforms=val_transforms,
    )
    embeddings = dataset / "validation" / cfg.datamodule.datasets.lang_dataset.lang_folder / "embeddings.npy"
    if not embeddings.is_file():
        raise FileNotFoundError(f"Missing HULC language embeddings: {embeddings}")
    model_cls = locate(cfg.model._target_)
    model = model_cls.load_from_checkpoint(str(checkpoint), map_location="cpu")
    model.load_lang_embeddings(embeddings)
    model.freeze()
    if cfg.model.action_decoder.get("load_action_bounds", False):
        raise ValueError("This inference loader expects the released model with saved action bounds")
    model = model.to(torch.device(device))
    model.eval()
    model.requires_grad_(False)
    conf = Path(calvin_agent.__file__).resolve().parents[1] / "conf"
    oracle = hydra.utils.instantiate(
        OmegaConf.load(conf / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    )
    env = PortableCalvinEnv(inputs, torch.device(device), renderer=renderer, show_gui=show_gui)
    return model, env, oracle, checkpoint
