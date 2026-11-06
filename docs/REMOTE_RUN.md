# Running WEAVE on your NVIDIA host

Run these checks **inside the SSH session** to identify the execution host.

```bash
uname -sm
nvidia-smi -L
```

For the pinned audit stack, expect `Linux x86_64` and an NVIDIA GPU listed by
`nvidia-smi`. An SSH client can run on macOS. If the target prints `Darwin`, use
the local inventory workflow there and run the physics audit on a supported
Linux host. The pinned SAPIEN release publishes Linux x86-64 wheels, and
ManiSkill's GPU physics requires Linux with NVIDIA.
[SAPIEN release files](https://pypi.org/project/sapien/3.0.0b1/#files),
[ManiSkill system support](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#system-support).

The execution host also needs Vulkan, Git, and Python 3.11 with `venv`.
On a Debian-based machine, install any missing graphics libraries once.

```bash
sudo apt update
sudo apt install libgl1 libglvnd0 libegl1-mesa libgles2-mesa libopengl0 vulkan-tools
nvidia-smi
vulkaninfo --summary
```

From the repository root, run setup and collect the complete Phase 0 audit.

```bash
./scripts/remote_gpu_setup.sh && make phase0-audit
```

Setup uses a dedicated `.venv-weave`, fetches the pinned RoboFactory source
and tabletop assets, and checks that PyTorch sees CUDA. The audit downloads
missing demo files and verifies them against the original manifest. Existing
verified files in `data/robofactory` are reused. The audit runs native CPU replay
on the training split and measures the native tabletop scenes on the selected
GPU. Select a single device with `CUDA_VISIBLE_DEVICES` when benchmarking a host
with several GPUs. Set `WEAVE_SETUP_PYTHON` to a compatible interpreter if it
has a different name.

The runner saves data inventory, per-episode replay evidence, benchmark
measurements, and a generated report under `results/phase0`. If an individual
simulation run fails, the remaining configurations still run and the final
exit status signals the error. Keep the JSON and the Markdown report together
when returning the artifacts.

## Running from a workstation

Start with a Git checkout at the same revision on the remote machine. You can
sync WEAVE working files and fetch its results over SSH.

```bash
make sync WEAVE_REMOTE=my-gpu
ssh my-gpu 'cd arm-skill-blender && ./scripts/remote_gpu_setup.sh'
make audit-remote WEAVE_REMOTE=my-gpu
make fetch-results WEAVE_REMOTE=my-gpu
```

Set `WEAVE_REMOTE_DIR` if the checkout lives elsewhere. Results fetched from
another source or configuration should go into a separate directory to keep
reports reproducible.

## Local inventory

A Mac can verify the recordings and render the audit report with no simulator
installation. Install the repository's `weave` extra in the local environment.

```bash
python -m pip install -e '.[weave]'
python -m weave.cli phase0-audit --mode inventory
python -m weave.cli report --output results/phase0/report.md
```

## Phase 1 smoke test

Use the environment from the completed Phase 0 run. Sync the current WEAVE files
from your workstation and connect to the host. In the remote checkout, check
availability with `nvtop` and set `CUDA_VISIBLE_DEVICES` to one free card.

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=6  # Use 6 if it is still free.
export MS_ASSET_DIR="/ICONAS/users/$USER/weave/maniskill"
export WEAVE_PHASE1_RESULTS="/ICONAS/users/$USER/weave/phase1"
make phase1-smoke
```

The runner downloads the ManiSkill YCB pack into the asset directory if needed.
It uses the existing `.venv-weave` and Phase 0 RoboFactory assets. The output
directory contains `smoke.json` and `rollout.jsonl` for review. The smoke test
uses random controls and does not train a model.
