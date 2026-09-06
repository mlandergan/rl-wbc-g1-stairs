# rl-wbc-g1-stairs

Vision-based stair climbing for the Unitree G1: a depth camera plus an AMP style reward, trained
with PPO in Isaac Lab. Part of the `rl-wbc-g1-*` series
([baseline](https://github.com/mlandergan/rl-wbc-g1-baseline) →
[amp](https://github.com/mlandergan/rl-wbc-g1-amp) → this).

The robot spawns on the flat border of a procedural pyramid-stairs tile and is commanded toward
the raised centre platform. Terrain difficulty is a curriculum: step height runs from 5 cm at
level 0 to 20 cm at level 9, and envs are promoted or demoted on how well they track their
commanded velocity.

## Requirements

A CUDA GPU box with Docker and an [NGC](https://catalog.ngc.nvidia.com/) account. This project's
own runs use a GCP L4 (`scripts/gcp_create_vm.sh` provisions one). Isaac Sim will not run on a
Mac and there is no CPU fallback.

## Build

```bash
docker login nvcr.io
docker build -f docker/Dockerfile -t rl-wbc-g1-stairs .
```

On a fresh VM, `scripts/remote_setup.sh` does the clone and build in one step. Rebuild only when
the Dockerfile changes — `isaaclab_project/` and `scripts/` are mounted at run time, so code edits
take effect without one.

## Train

```bash
# no video
./scripts/train_amp_depth.sh

# with video: train to completion, then export a clip per checkpoint
./scripts/train_amp_depth.sh && ./scripts/export_checkpoint_videos.sh
```

Headless, 4096 envs, using the budget in `isaaclab_project/g1_stairs/agents/skrl_g1_stairs_cfg.yaml`
(240,000 timesteps = 10,000 PPO iterations). Checkpoints and TensorBoard logs land in
`logs/skrl/g1_stairs/<timestamp>_ppo_torch/`; `scripts/sync_results.sh` pulls them back.

For a long run, detach it: `nohup ./scripts/train_amp_depth.sh > /tmp/train.log 2>&1 &`

Useful overrides: `NUM_ENVS=2048 ./scripts/train_amp_depth.sh`,
`NUM_ENVS=8 VIDEO_LENGTH=400 ./scripts/export_checkpoint_videos.sh`, and
`./scripts/export_checkpoint_videos.sh <run_dir>` to export from an older run.

> **Record video after training, not during it.** Training at 4096 envs alongside a video job
> with `--enable_cameras` OOM-killed a run on 2026-08-12, and hung on 2026-08-13. If you want
> clips as they appear anyway, `scripts/watch_checkpoints.sh` polls for new checkpoints with a
> per-export timeout — run training at a reduced `NUM_ENVS` alongside it.

## What to watch

`diag_mean_terrain_level` in TensorBoard is the scalar that matters: mean terrain difficulty
across envs, starting at 2. If it climbs, the curriculum is working. If it stays flat, nothing
else is worth reading yet. Supporting: `diag_mean_planar_dist_to_goal` (down),
`diag_mean_terrain_height_at_root` (up), `rew_dont_wait` (toward 0).

Startup also prints measured per-level platform heights, which is the cheapest check that the
generated stair geometry is what the config thinks it is.

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
