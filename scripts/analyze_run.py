#!/usr/bin/env python3
"""Summarize a g1_stairs training run from its TensorBoard event file.

Defaults to the most recent run under logs/skrl/g1_stairs/ (by directory mtime), so
during training you can just run it with no arguments.

    ./scripts/analyze_run.sh                       # latest run
    ./scripts/analyze_run.sh --baseline 2026-09-16_17-12-17_wasabi_amp_depth
    ./scripts/analyze_run.sh --run <dir> --at 66600

    # every rew_* term, with columns at earlier checkpoints:
    ./scripts/analyze_run.sh --rewards --cols 30300,60100 \
        --baseline 2026-09-16_17-12-17_wasabi_amp_depth

Why a baseline matters: on 2026-09-21 a run posted HIGHER reward than a known-good
run while climbing 21x less (feet_air_time reward hacking). Absolute numbers hid it;
the side-by-side at matched steps made it obvious. --baseline compares at the newer
run's latest step, so you are never comparing different points on the curve.

Why --cols matters: a single snapshot cannot distinguish "healthy and converged"
from "degrading slowly". Columns at earlier steps turn each reward term into a
trajectory, which is what actually caught the air-time exploit ramping.

Read-only: opens event files, writes nothing.
"""
import argparse, datetime, glob, os, re, sys
import numpy as np

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    sys.exit("tensorboard not importable. Use the venv:\n  tensor-env/bin/python scripts/analyze_run.py")

LOGROOT = "logs/skrl/g1_stairs"
REF_TAG = "Reward / Instantaneous reward (mean)"


# Reward terms where "bigger" is not automatically "better". Everything else gets
# good_direction=+1: for a positive term more reward is better, and for a penalty
# term rising toward zero means the penalty is shrinking, which is also better.
# feet_air_time is the known exploit surface, so its direction is deliberately
# left neutral -- a rising air-time reward is the signature we are watching for.
NEUTRAL_REWARD_TERMS = {"feet_air_time"}

# (label, tag, good_direction) -- good_direction: +1 higher better, -1 lower better, 0 neutral
GROUPS = {
    "TASK PROGRESS": [
        ("climb (m, episode max)", "Info / diag_mean_episode_max_climb_m", +1),
        ("terrain level",          "Info / diag_mean_terrain_level",       +1),
        ("climb fraction",         "Info / diag_curriculum_climb_fraction", +1),
        ("reached goal frac",      "Info / diag_frac_reached_goal",        +1),
        ("planar dist to goal (m)","Info / diag_mean_planar_dist_to_goal", -1),
        ("episode len (steps)",    "Episode / Total timesteps (mean)",     +1),
    ],
    "CURRICULUM": [
        ("move UP frac",   "Info / diag_curriculum_move_up_frac",   +1),
        ("move DOWN frac", "Info / diag_curriculum_move_down_frac", -1),
        ("track score",    "Info / diag_curriculum_track_score",    +1),
    ],
    "REWARD (key terms)": [
        ("reward/step (mean)",  "Reward / Instantaneous reward (mean)", 0),
        ("  feet_air_time",     "Info / rew_feet_air_time",             0),
        ("  task_lin_vel",      "Info / rew_task_lin_vel",             +1),
        ("  task_ang_vel",      "Info / rew_task_ang_vel",             +1),
        ("  is_alive",          "Info / rew_is_alive",                 +1),
        ("  height_progress",   "Info / rew_height_progress",          +1),
        ("  dont_wait",         "Info / rew_dont_wait",                +1),
        ("  edge_penetration",  "Info / rew_edge_penetration",         +1),
        ("  flat_orientation",  "Info / rew_flat_orientation_l2",      +1),
    ],
    "AMP / STYLE": [
        ("style share",         "AMP / style share",          +1),
        ("style reward (raw)",  "AMP / style reward (raw)",   +1),
        ("discriminator logit", "AMP / discriminator logit",   0),
        ("discriminator loss",  "Loss / Discriminator loss",   0),
    ],
    "OPTIMIZATION": [
        ("value loss",    "Loss / Value loss",           -1),
        ("policy loss",   "Loss / Policy loss",           0),
        ("entropy loss",  "Loss / Entropy loss",          0),
        ("policy std",    "Policy / Standard deviation",  0),
        ("learning rate", "Learning / Learning rate",     0),
    ],
    "TERMINATIONS (per step)": [
        ("height", "Info / diag_term_height",  -1),
        ("contact","Info / diag_term_contact", -1),
        ("tilt",   "Info / diag_term_tilt",    -1),
    ],
}


def read_agent_params(run_dir):
    """Pull rollouts / total timesteps / checkpoint_interval from the run's own config.

    skrl counts `timesteps` in ENV steps and performs one PPO update per `rollouts`
    steps, so iterations = timesteps / rollouts. Reading this from the run's
    params/agent.yaml rather than hardcoding 24/240000 means the iteration column
    stays honest if a future run changes either value. Parsed with a regex because
    the eval venv has no PyYAML and the fields are flat scalars.
    """
    out = {"rollouts": None, "timesteps": None, "checkpoint_interval": None}
    path = os.path.join(run_dir, "params", "agent.yaml")
    if not os.path.isfile(path):
        return out
    pats = {
        "rollouts": r"^\s*rollouts:\s*(\d+)",
        "timesteps": r"^\s*timesteps:\s*(\d+)",           # not random_timesteps: anchored at line start
        "checkpoint_interval": r"^\s*checkpoint_interval:\s*(\d+)",
    }
    try:
        with open(path) as fh:
            for line in fh:
                for key, pat in pats.items():
                    if out[key] is None:
                        m = re.match(pat, line)
                        if m:
                            out[key] = int(m.group(1))
    except OSError:
        pass
    return out


def latest_checkpoint(run_dir):
    """(basename, step) of the highest-numbered agent_*.pt, plus how many exist."""
    ckpts = glob.glob(os.path.join(run_dir, "checkpoints", "agent_*.pt"))
    if not ckpts:
        return None, None, 0
    best, best_step = None, -1
    for c in ckpts:
        m = re.search(r"agent_(\d+)\.pt$", os.path.basename(c))
        if m and int(m.group(1)) > best_step:
            best, best_step = os.path.basename(c), int(m.group(1))
    n_all = len(glob.glob(os.path.join(run_dir, "checkpoints", "*.pt")))
    return best, best_step, n_all


def throughput(ea, npts=60):
    """(steps_per_min, last_wall_time) measured over the most recent `npts` points."""
    try:
        s = ea.Scalars(REF_TAG)
    except KeyError:
        return None, None
    if len(s) < 2:
        return None, (s[-1].wall_time if s else None)
    w = s[-npts] if len(s) > npts else s[0]
    dt = s[-1].wall_time - w.wall_time
    if dt <= 0:
        return None, s[-1].wall_time
    return (s[-1].step - w.step) / dt * 60.0, s[-1].wall_time


def load(run_dir):
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    tags = set(ea.Tags()["scalars"])
    return ea, tags


def series(ea, tags, tag):
    if ea is None or tag not in tags:
        return None
    s = ea.Scalars(tag)
    return np.array([(x.step, x.value) for x in s], dtype=float)


def window(arr, at=None, frac=0.10, minpts=5):
    """Mean over the last `frac` of points at or before step `at`."""
    if arr is None or len(arr) == 0:
        return float("nan")
    a = arr if at is None else arr[arr[:, 0] <= at]
    if len(a) == 0:
        return float("nan")
    n = max(minpts, int(len(a) * frac))
    return float(a[-n:, 1].mean())


def fmt(v, width=11):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return " " * (width - 1) + "-"
    return f"{v:{width}.4g}"


def human_step(s):
    return f"@{s / 1000:.1f}k" if s >= 1000 else f"@{int(s)}"


def trend_arrow(arr, good):
    """Compare last 10% against the preceding 10%."""
    if arr is None or len(arr) < 20:
        return "  "
    n = max(5, int(len(arr) * 0.10))
    recent, prior = arr[-n:, 1].mean(), arr[-2 * n:-n, 1].mean()
    if prior == 0 and recent == 0:
        return "= "
    denom = abs(prior) if prior else 1.0
    delta = (recent - prior) / denom
    if abs(delta) < 0.05:
        return "= "
    up = delta > 0
    if good == 0:
        return "^ " if up else "v "
    improving = (up and good > 0) or (not up and good < 0)
    return ("^ " if up else "v ") + ("OK" if improving else "!!")


def print_reward_table(ea, tags, cols, base_ea, base_tags, base_at=None):
    """Every Info / rew_* term, at several step columns, with share of budget.

    The logged rew_* scalars are already weight-multiplied, so they sum to the
    logged instantaneous reward. The NET/residual rows at the bottom assert that,
    which is what makes the SHARE column trustworthy rather than indicative --
    if a term were missing from the event file the residual would expose it.
    """
    last_step = cols[-1]
    base_at = last_step if base_at is None else base_at
    rew_tags = sorted(t for t in tags if t.startswith("Info / rew_"))
    if not rew_tags:
        print("  (no 'Info / rew_*' tags in this event file)")
        return

    rows, inactive = [], []
    for t in rew_tags:
        arr = series(ea, tags, t)
        vals = [window(arr, c) for c in cols]
        now = vals[-1]
        name = t.replace("Info / rew_", "")
        if np.isnan(now):
            continue
        if abs(now) < 1e-12 and (arr is None or not np.any(np.abs(arr[:, 1]) > 1e-12)):
            inactive.append(name)
            continue
        good = 0 if name in NEUTRAL_REWARD_TERMS else +1
        rows.append({
            "name": name,
            "vals": vals,
            "now": now,
            "lo": float(arr[:, 1].min()),
            "hi": float(arr[:, 1].max()),
            "trend": trend_arrow(arr, good),
            "base": window(series(base_ea, base_tags, t), base_at),
        })

    pos_total = sum(r["now"] for r in rows if r["now"] > 0) or 1.0
    neg_total = sum(-r["now"] for r in rows if r["now"] < 0) or 1.0

    col_hdr = "".join(f"{human_step(c):>11s}" for c in cols)
    hdr = f"  {'TERM':26s}{col_hdr}{'MIN':>11s}{'MAX':>11s}{'TREND':>7s}"
    if base_ea:
        hdr += f"{'BASELINE':>11s}{'RATIO':>9s}"
    hdr += f"{'SHARE':>8s}"
    width = len(hdr)

    print("-" * width)
    print("FULL REWARD TABLE  --  all Info / rew_* terms, already weighted (they sum to reward/step)")
    print(hdr)

    def emit(section, subset):
        if not subset:
            return 0.0
        print(f"  {section}")
        subtotal = 0.0
        for r in sorted(subset, key=lambda x: -abs(x["now"])):
            subtotal += r["now"]
            share = (r["now"] / pos_total * 100) if r["now"] > 0 else (r["now"] / neg_total * 100)
            line = f"    {r['name']:24s}" + "".join(fmt(v) for v in r["vals"])
            line += fmt(r["lo"]) + fmt(r["hi"]) + f"{r['trend']:>7s}"
            if base_ea:
                ratio = ""
                if not np.isnan(r["base"]) and abs(r["base"]) > 1e-9:
                    ratio = f"{r['now'] / r['base']:8.2f}x"
                line += fmt(r["base"]) + f"{ratio:>9s}"
            line += f"{share:+7.1f}%"
            print(line)
        print(f"    {'subtotal':24s}{'':>{11 * (len(cols) - 1)}}{subtotal:11.4g}")
        return subtotal

    pos_sum = emit("POSITIVE", [r for r in rows if r["now"] > 0])
    neg_sum = emit("NEGATIVE", [r for r in rows if r["now"] < 0])

    # ---- completeness assertion -----------------------------------------
    net = pos_sum + neg_sum
    logged = window(series(ea, tags, "Reward / Instantaneous reward (mean)"), last_step)
    print("-" * width)
    print(f"  {'NET (sum of all terms)':26s}{'':>{11 * (len(cols) - 1)}}{net:11.4g}")
    print(f"  {'logged reward/step':26s}{'':>{11 * (len(cols) - 1)}}{logged:11.4g}")
    if not np.isnan(logged):
        resid = net - logged
        flag = "ok" if abs(resid) < max(0.02, abs(logged) * 0.01) else "MISMATCH"
        print(f"  {'residual':26s}{'':>{11 * (len(cols) - 1)}}{resid:11.4g}   [{flag}]")
    if inactive:
        print(f"  inactive (identically zero, weight disabled): {', '.join(inactive)}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", help="run dir (default: most recent by mtime)")
    p.add_argument("--baseline", help="run dir to compare against, at the same step")
    p.add_argument("--at", type=int, help="evaluate at this step instead of latest")
    p.add_argument("--rewards", action="store_true", help="print the full per-term reward table")
    p.add_argument("--cols", help="comma-separated earlier steps to add as columns "
                                  "(default: quarter and half of the latest step)")
    p.add_argument("--logroot", default=LOGROOT)
    args = p.parse_args()

    runs = sorted(glob.glob(os.path.join(args.logroot, "*/")), key=os.path.getmtime)
    if not runs:
        sys.exit(f"no runs under {args.logroot}")
    run = args.run or runs[-1]
    if not os.path.isdir(run):
        cand = os.path.join(args.logroot, run)
        run = cand if os.path.isdir(cand) else sys.exit(f"no such run: {run}")

    ea, tags = load(run)
    ref = series(ea, tags, "Reward / Instantaneous reward (mean)")
    if ref is None or not len(ref):
        sys.exit(f"no scalar data yet in {run} (training may still be booting)")
    last_step = int(args.at or ref[-1, 0])

    if args.cols:
        cols = [int(float(c)) for c in args.cols.split(",") if c.strip()]
    else:
        cols = [last_step // 4, last_step // 2]
    cols = sorted({c for c in cols if 0 < c < last_step}) + [last_step]

    base_ea = base_tags = None
    base_at = last_step
    base_max = None
    if args.baseline:
        b = args.baseline if os.path.isdir(args.baseline) else os.path.join(args.logroot, args.baseline)
        if not os.path.isdir(b):
            sys.exit(f"no such baseline: {args.baseline}")
        base_ea, base_tags = load(b)
        # The whole point of --baseline is a step-MATCHED comparison. If the baseline
        # run is shorter than the current step, window() would silently fall back to
        # the baseline's final values and the ratios would quietly stop being
        # step-matched. Clamp explicitly and say so, rather than hiding it.
        bref = series(base_ea, base_tags, REF_TAG)
        if bref is not None and len(bref):
            base_max = int(bref[-1, 0])
            base_at = min(last_step, base_max)


    print("=" * 96)
    print(f"RUN      : {os.path.basename(run.rstrip('/'))}")

    prm = read_agent_params(run)
    total_steps = prm["timesteps"]
    rollouts = prm["rollouts"]

    step_line = f"STEP     : {last_step:,}"
    if total_steps:
        step_line += f" / {total_steps:,}  ({last_step / total_steps * 100:.1f}%)"
    step_line += f"   [{len(ref)} logged points]"
    print(step_line)

    # skrl does one PPO update per `rollouts` env steps, so iteration == step / rollouts.
    if rollouts:
        it_now = last_step // rollouts
        it_line = f"ITER     : {it_now:,}"
        if total_steps:
            it_line += f" / {total_steps // rollouts:,}"
        it_line += f"   (rollouts={rollouts}, so 1 iter = {rollouts} env steps)"
        print(it_line)

    ck_name, ck_step, ck_count = latest_checkpoint(run)
    if ck_name:
        ck_line = f"CKPT     : {ck_name}"
        if rollouts:
            ck_line += f"  = iter {ck_step // rollouts:,}"
        ck_line += f"   [{ck_count} saved"
        if prm["checkpoint_interval"] and rollouts:
            ck_line += f", every {prm['checkpoint_interval'] // rollouts} iters"
        ck_line += "]"
        print(ck_line)

    rate, wall = throughput(ea)
    # A stopped run still has a rate, but extrapolating it is meaningless -- the
    # 09-16 baseline ended at 93.6k and its trailing wall-clock produced a
    # "479 h remaining" ETA. Only project forward if the event file is still live.
    stale_s = (datetime.datetime.now().timestamp() - wall) if wall else None
    is_live = stale_s is not None and stale_s < 600
    if rate and total_steps and last_step < total_steps and not args.at and is_live:
        remain = (total_steps - last_step) / rate  # minutes
        # .astimezone() on a naive datetime attaches the local zone, which is what
        # makes %Z render "PDT" instead of an empty string.
        eta = datetime.datetime.fromtimestamp(wall + remain * 60.0).astimezone()
        print(f"RATE     : {rate:,.0f} steps/min   ->  ETA {eta:%a %I:%M %p %Z}"
              f"   [{remain / 60.0:.2f} h remaining]")
    elif stale_s is not None:
        ended = datetime.datetime.fromtimestamp(wall).astimezone()
        state = "stopped" if (total_steps and last_step < total_steps) else "complete"
        print(f"STATE    : {state}, last write {ended:%Y-%m-%d %I:%M %p %Z} "
              f"({stale_s / 3600.0:.1f} h ago)")


    if args.baseline:
        print(f"BASELINE : {os.path.basename(args.baseline.rstrip('/'))}  "
              f"(sampled at step {base_at:,})")
        if base_max is not None and base_at < last_step:
            print(f"  !! NOT STEP-MATCHED: baseline run ends at {base_max:,}, current run is at "
                  f"{last_step:,}.")
            print(f"     BASELINE/RATIO columns compare against that run's FINAL state, not the "
                  f"same point on the curve.")
    print("=" * 96)


    hdr = f"{'METRIC':30s} {'NOW':>11s} {'MIN':>11s} {'MAX':>11s} {'TREND':>6s}"
    if base_ea:
        hdr += f" {'BASELINE':>11s} {'RATIO':>8s}"
    print(hdr)

    for group, items in GROUPS.items():
        print("-" * 96)
        print(group)
        for label, tag, good in items:
            arr = series(ea, tags, tag)
            now = window(arr, last_step)
            lo = float(arr[:, 1].min()) if arr is not None and len(arr) else float("nan")
            hi = float(arr[:, 1].max()) if arr is not None and len(arr) else float("nan")
            row = f"  {label:28s} {fmt(now)} {fmt(lo)} {fmt(hi)} {trend_arrow(arr, good):>6s}"
            if base_ea:
                barr = series(base_ea, base_tags, tag)
                bv = window(barr, base_at)
                ratio = ""
                if not np.isnan(bv) and not np.isnan(now) and abs(bv) > 1e-9:
                    r = now / bv
                    ratio = f"{r:7.2f}x"
                row += f" {fmt(bv)} {ratio:>8s}"
            print(row)

    # ---- full per-term reward table --------------------------------------
    if args.rewards:
        print_reward_table(ea, tags, cols, base_ea, base_tags, base_at)

    # ---- reward budget ---------------------------------------------------
    print("-" * 96)
    print("REWARD BUDGET (share of positive / negative income per step)")
    terms = [(t.replace("Info / rew_", ""), window(series(ea, tags, t), last_step))
             for t in sorted(tags) if t.startswith("Info / rew_")]
    terms = [(k, v) for k, v in terms if not np.isnan(v) and abs(v) > 1e-9]
    pos = sum(v for _, v in terms if v > 0) or 1.0
    neg = sum(-v for _, v in terms if v < 0) or 1.0
    for k, v in sorted(terms, key=lambda kv: -abs(kv[1]))[:12]:
        share = (v / pos * 100) if v > 0 else (v / neg * 100)
        bar = "#" * min(40, int(abs(share) / 2.5))
        print(f"  {k:28s} {v:11.4g}  {share:+6.1f}%  {bar}")

    # ---- verdicts --------------------------------------------------------
    print("-" * 96)
    print("HEALTH CHECKS")
    g = lambda t: window(series(ea, tags, t), last_step)
    checks = []
    up = g("Info / diag_curriculum_move_up_frac")
    checks.append((not np.isnan(up) and up > 1e-5, f"curriculum promoting (move_up_frac={up:.3g}, want >0)"))
    climb = g("Info / diag_mean_episode_max_climb_m")
    checks.append((not np.isnan(climb) and climb > 0.10, f"climbing (max_climb={climb:.3g} m, want >0.10)"))
    air, lin = g("Info / rew_feet_air_time"), g("Info / rew_task_lin_vel")
    checks.append((not np.isnan(air) and not np.isnan(lin) and air < lin,
                   f"air_time not dominating (air={air:.3g} vs lin_vel={lin:.3g})"))
    share = g("AMP / style share")
    checks.append((not np.isnan(share) and share > 0.01, f"AMP style contributing (share={share:.3g}, want >0.01)"))
    eplen = g("Episode / Total timesteps (mean)")
    checks.append((not np.isnan(eplen) and eplen > 400, f"episodes surviving (len={eplen:.0f}, want >400 of 1000)"))
    for ok, msg in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    print("=" * 96)


if __name__ == "__main__":
    main()
