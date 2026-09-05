#!/usr/bin/env python3
"""Hook our task's gym registration into Isaac Lab's train.py / play.py entrypoints, at the
exact point Isaac Lab registers its own built-in tasks: `isaaclab_project.g1_stairs` (vision-
based AMP stair climbing, forked from rl-wbc-g1-amp's g1_amp, trained via skrl) into the skrl
entrypoints. No rsl_rl hook here -- unlike rl-wbc-g1-amp (which also carries a PPO-only
g1_baseline task for reference), this project's isaaclab_project/ only has g1_stairs.

Verified against Isaac Lab's actual scripts/reinforcement_learning/skrl/train.py, which imports
`isaaclab_tasks` (side-effect only, for gym.register calls) right after AppLauncher starts the
simulation app and before the @hydra_task_config main(). This is the same insertion point Isaac
Lab's own external-extension template uses for third-party tasks. Idempotent (safe to run more
than once) and does not assume a specific install path -- it searches the image's filesystem for
the target files instead of hardcoding one.
"""

import pathlib
import subprocess
import sys

ANCHOR = "import isaaclab_tasks  # noqa: F401"

HOOKS = {
    "skrl": "import isaaclab_project.g1_stairs  # noqa: F401 (stairs AMP task registration)",
}


def find_targets(library: str) -> list[pathlib.Path]:
    # -xdev keeps this from wandering into /proc, /sys, or other mounted filesystems.
    out = subprocess.run(
        ["find", "/", "-xdev", "-path", f"*/reinforcement_learning/{library}/train.py",
         "-o", "-path", f"*/reinforcement_learning/{library}/play.py"],
        capture_output=True, text=True,
    )
    return [pathlib.Path(p) for p in out.stdout.splitlines() if p]


def main() -> None:
    any_found = False
    for library, hook_line in HOOKS.items():
        targets = find_targets(library)
        if not targets:
            print(
                f"WARNING: could not find Isaac Lab's {library} train.py/play.py anywhere on "
                f"this filesystem -- skipping. If that library isn't installed in this image, "
                f"this is expected; otherwise patch by hand: add\n"
                f"    {hook_line}\n"
                f"immediately after the line `{ANCHOR}` in {library}/train.py and play.py."
            )
            continue
        any_found = True
        for path in targets:
            text = path.read_text()
            if hook_line in text:
                print(f"already patched: {path}")
                continue
            if ANCHOR not in text:
                print(f"WARNING: anchor line not found in {path} -- skipping (Isaac Lab version mismatch?)")
                continue
            path.write_text(text.replace(ANCHOR, f"{ANCHOR}\n{hook_line}", 1))
            print(f"patched: {path}")

    if not any_found:
        sys.exit("skrl's train.py/play.py were not found. Is Isaac Lab actually installed?")


if __name__ == "__main__":
    main()
