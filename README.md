# rl-wbc-g1-stairs

Vision-based stair climbing for the Unitree G1: a depth camera plus an AMP style reward, trained
with PPO in Isaac Lab. Part of the `rl-wbc-g1-*` series
([baseline](https://github.com/mlandergan/rl-wbc-g1-baseline) →
[amp](https://github.com/mlandergan/rl-wbc-g1-amp) → this).

The robot spawns on the flat border of a procedural pyramid-stairs tile and is commanded toward
the raised centre platform. Terrain difficulty is a curriculum: step height runs from 5 cm at
level 0 to 20 cm at level 9, and envs are promoted or demoted on how far up the stairs they
actually get (climb progress and goal arrival).

## Requirements

A CUDA GPU box with Docker and an [NGC](https://catalog.ngc.nvidia.com/) account. Isaac Sim will
not run on a Mac and there is no CPU fallback.

- **Cloud** — the Quickstart below provisions a GCP L4, which is what this project's own runs use.
- **Your own machine** — Ubuntu with an NVIDIA GPU works too; skip to
  [Running locally](#running-locally-ubuntu--nvidia-gpu).

---

## Quickstart

### 1. Get a GPU box

```bash
PROJECT_ID=<your-gcp-project> ZONE=us-east1-b ./scripts/gcp_create_vm.sh
gcloud compute ssh g1-stairs-l4 --zone=us-east1-b --project=<your-gcp-project>
```

Then on the VM:

```bash
git clone https://github.com/mlandergan/rl-wbc-g1-stairs.git ~/rl-wbc-g1-stairs
cd ~/rl-wbc-g1-stairs && git submodule update --init --recursive
```

> **Check the GPU driver before going further.** `nvidia-smi` must work. If it doesn't, the
> `install-nvidia-driver=True` metadata flag did nothing (it often doesn't on plain Ubuntu) —
> install it manually, or create the VM from a Deep Learning VM image instead.

### 2. Build the image

```bash
sudo docker login nvcr.io          # username: $oauthtoken, password: your NGC API key
sudo docker build -f docker/Dockerfile -t rl-wbc-g1-stairs .
```

`scripts/remote_setup.sh` wraps steps 1–2 (Docker + NVIDIA Container Toolkit + build) if you'd
rather run one script.

Rebuild only when the Dockerfile changes — `isaaclab_project/` and `scripts/` are mounted at run
time, so code edits take effect without one.

> **If you plan to record video, do this too.** Compute-only drivers (Deep Learning VM images,
> any `-server` driver package) ship without the Vulkan/OpenGL libraries Isaac Sim's renderer
> needs. Video jobs then *hang* rather than erroring. Check with `dpkg -l | grep nvidia` — if you
> see `-server`-suffixed packages, run:
> ```bash
> sudo apt-get install -y libnvidia-gl-<version>-server libgl1 libglvnd0 libglx0 vulkan-tools
> sudo nvidia-ctk cdi generate --output=/var/run/cdi/nvidia.yaml   # required: the CDI spec is cached
> vulkaninfo --summary | grep -A2 GPU0                              # should list your NVIDIA GPU
> ```

### 3. Train

```bash
./scripts/train_amp_depth.sh
```

Headless, 4096 envs, using the budget in
`isaaclab_project/g1_stairs/agents/skrl_g1_stairs_cfg.yaml` (240,000 timesteps = 10,000 PPO
iterations). Output lands in `logs/skrl/g1_stairs/<timestamp>_wasabi_amp_depth/`.

For a long run, detach it so an SSH drop doesn't kill it:

```bash
nohup ./scripts/train_amp_depth.sh > /tmp/train.log 2>&1 &
```

Common variations:

```bash
NUM_ENVS=2048 ./scripts/train_amp_depth.sh                    # smaller GPU
./scripts/train_amp_depth.sh --max_iterations 500             # short run
./scripts/train_amp_depth.sh --checkpoint <container-path>    # resume from a checkpoint
```

### 4. Record videos

**From specific checkpoints** (the usual case — pick the interesting ones):

```bash
RUN=logs/skrl/g1_stairs/2026-09-11_01-39-28_wasabi_amp_depth

sudo docker run --rm --gpus all -e ACCEPT_EULA=Y \
  -v "$(pwd)/isaaclab_project:/workspace/rl_wbc_g1/isaaclab_project" \
  -v "$(pwd)/scripts:/workspace/rl_wbc_g1/scripts" \
  -v "$(pwd)/logs:/workspace/isaaclab_root/logs" \
  rl-wbc-g1-stairs \
  bash -lc "cd /workspace/isaaclab_root && ./isaaclab.sh -p /workspace/rl_wbc_g1/scripts/play_amp_depth.py \
    --checkpoint /workspace/isaaclab_root/${RUN}/checkpoints/agent_240000.pt \
    --num_envs 4 --video_length 300 --headless"
```

Checkpoint filenames are in **timesteps**, not iterations: `agent_<iteration × 24>.pt`
(`rollouts: 24`). So iteration 1500 → `agent_36000.pt`, iteration 10000 → `agent_240000.pt`.
Videos land in `<run>/videos/play_agent_<N>/`.

**From every checkpoint in a run:**

```bash
./scripts/export_checkpoint_videos.sh                  # most recent run
./scripts/export_checkpoint_videos.sh <run_dir>        # a specific one
NUM_ENVS=8 VIDEO_LENGTH=400 ./scripts/export_checkpoint_videos.sh
```

> **Record after training, not during it.** Two Isaac Sim processes on one GPU OOM-killed a run
> (2026-08-12) and hung another (2026-08-13). Stop training first. To capture clips as training
> progresses anyway, `train_amp_depth.sh --video` records in-process through the same Isaac Sim
> instance, which avoids that conflict (`--video_interval` controls spacing).

### 5. Pull results down and shut off

```bash
./scripts/sync_results.sh                                       # from your local machine
gcloud compute instances stop g1-stairs-l4 --zone=us-east1-b --project=<your-gcp-project>
```

`./scripts/gcp_autostop.sh` arms a detached local timer that stops the VM after a fixed delay
(`DELAY_MINUTES=120 ./scripts/gcp_autostop.sh`) — worth running at the *start* of a session, not
as an afterthought. A GPU VM left running is the single easiest way to waste money here.

---

## Running locally (Ubuntu + NVIDIA GPU)

No cloud needed. You need Ubuntu, a working NVIDIA driver (`nvidia-smi` must print your GPU), and
an NGC account. Steps 3–4 of the Quickstart are identical — only the setup differs.

### One-time setup

```bash
git clone https://github.com/mlandergan/rl-wbc-g1-stairs.git && cd rl-wbc-g1-stairs
git submodule update --init --recursive

# Docker + NVIDIA Container Toolkit
curl -fsSL https://get.docker.com | sudo sh
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker

sudo usermod -aG docker "$USER"   # then log out and back in
```

The `docker` group matters: `train_amp_depth.sh` calls `docker` without `sudo`, so it only works
once your user is in that group. (`export_checkpoint_videos.sh` uses `sudo docker` and works
either way.)

Verify the GPU is visible inside a container before building:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

### Build

```bash
docker login nvcr.io          # username: $oauthtoken, password: your NGC API key
docker build -f docker/Dockerfile -t rl-wbc-g1-stairs .
```

> The Vulkan/OpenGL gotcha in the Quickstart is mostly a **cloud** problem — normal desktop
> driver packages already include the GL libraries, while headless/`-server` ones don't. Confirm
> with `vulkaninfo --summary | grep -A2 GPU0`; if it lists your GPU, you're fine and can skip that
> fix entirely.

### Train and record

Same commands as Quickstart steps 3–4, without `sudo`. Size `NUM_ENVS` to your VRAM — measured
here, **4096 envs uses ~17 GB**, so:

| VRAM | Suggested `NUM_ENVS` |
|---|---|
| 24 GB (4090, 3090, L4) | 4096 |
| 12–16 GB | 2048 |
| 8–10 GB | 1024 |

```bash
NUM_ENVS=2048 ./scripts/train_amp_depth.sh
./scripts/export_checkpoint_videos.sh
```

Everything already writes to `logs/` on this machine, so `sync_results.sh` and `gcp_autostop.sh`
don't apply.

---

## What to watch

`diag_mean_terrain_level` in TensorBoard is the scalar that matters: mean terrain difficulty
across envs, starting at 2. If it climbs, the curriculum is working. If it stays flat, nothing
else is worth reading yet.

Supporting: `diag_curriculum_climb_fraction` (toward 1.0 = full summit),
`diag_frac_reached_goal` (up), `diag_mean_planar_dist_to_goal` (down),
`diag_mean_episode_max_climb_m` (up), `rew_dont_wait` (toward 0).

Startup also prints measured per-level platform heights — the cheapest check that the generated
stair geometry matches what the config thinks it is.

## Layout

| Path | What |
|---|---|
| `isaaclab_project/g1_stairs/` | env, config, models, AMP agent, motion loader |
| `scripts/` | Docker wrappers for train / eval / video, plus GCP VM lifecycle |
| `docker/` | image definition and task registration |
| `AMP_data/` | reference motion source and result media (read its README before trusting a video) |

## License

BSD-3-Clause (see `LICENSE`), matching upstream. The core environment is adapted from
[`linden713/humanoid_amp`](https://github.com/linden713/humanoid_amp) (BSD-3-Clause) via
`rl-wbc-g1-amp`. `THIRD_PARTY_NOTICES.md` has full attribution, including the
design-reference-only (never vendored) comparison against
[`project-instinct/InstinctLab`](https://github.com/project-instinct/InstinctLab) (CC BY-NC 4.0).
