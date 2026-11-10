# Aster GPU physics pilot

The Warp collector batches independent worlds of the same layout on one
CUDA device. MuJoCo Warp advances the physics. The existing expert,
inverse kinematics, panel interlock logic, and ray query run on the host.
The host ray query uses transforms produced by Warp. Each controller receives
observations at its original requested timestep, including individual panel
steps. CUDA graph capture reduces per-step launch overhead.

Warp uses a separate package environment and dataset contract. The original
MuJoCo 3.3.0 collector and saved corpus keep their original provenance.
The GPU command requires a successful local physics gate before writing
episodes. Its output is experimental and has its own controller hash.
The existing CPU-study trainer accepts the original controller hash.

## Copy and install

From the Mac repo root, copy the four GPU files to the existing Aster
checkout on the lab server.

```sh
rsync -av projects/single_arm_composition/larger_scale/{warp_sim.py,warp_collect.py,requirements-warp.txt,GPU.md} \
  rishabh@icon.ist.berkeley.edu:arm-skill-blender-aster/projects/single_arm_composition/larger_scale/
```

Check with Kartik about the extra packages. Then run on the lab server.

```sh
cd ~/arm-skill-blender-aster/projects/single_arm_composition
conda create -n aster-warp python=3.12 pip
conda activate aster-warp
python -m pip install --no-cache-dir -r larger_scale/requirements-warp.txt
```

## Select a GPU and run the pilot

Use `nvtop` to pick one idle GPU, then press Q. Change the example GPU
number below to the one you picked. The lab requires a message in
`#server` before using multiple GPUs.

```sh
export CUDA_VISIBLE_DEVICES=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
run="/ICONAS/datasets/$USER/aster-large"
gpu="/ICONAS/datasets/$USER/aster-warp-pilot"

python -m larger_scale.warp_collect check --out "$gpu/gate.json"
```

When the check prints `"passed": true`, run the pilot. It uses the same
sealed manifest and writes a separate corpus on the NAS.

```sh
python -m larger_scale.warp_collect collect \
  --manifest "$run/manifest" --gate "$gpu/gate.json" \
  --out "$gpu/train" --count 768 --worlds 32 --shard-size 768
```

First-use kernel compilation can take time. The printed seconds include
setup and compilation. The pilot is for measuring throughput and physical
success on the target GPU before allocating a larger run.
