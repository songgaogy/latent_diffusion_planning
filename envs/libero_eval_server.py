"""Server worker for the LIBERO eval bridge.

Runs in the LIBERO python selected by envs.libero_remote_env.

Communication: length-prefixed pickle frames over stdin (commands) / stdout
(responses). Stderr stays unbuffered for log drainage by the parent.

Init handshake (one frame from parent):
    {
        "task_suite": "libero_10",
        "task_id": int,
        "camera_heights": int, "camera_widths": int,
        "camera_names": list[str],
        "controller": "OSC_POSE",
        "horizon": int,
    }

Per-step protocol:
    parent -> {"cmd": "reset"}                       -> {"obs": {...}}
    parent -> {"cmd": "set_init_state", "idx": int}  -> {"obs": {...}}
    parent -> {"cmd": "step", "action": ndarray}     ->
        {"obs": {...}, "reward": float, "done": bool, "success": bool}
    parent -> {"cmd": "close"}                       -> {"ok": True}

obs dict is already in the dataset namespace (see libero_obs_adapter).
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import numpy as np

# Make the repo root importable so the libero subprocess can find
# envs.libero_proto / envs.libero_obs_adapter without installing this repo.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from envs.libero_obs_adapter import adapt_obs  # noqa: E402
from envs.libero_proto import read_frame, write_frame  # noqa: E402


def _stdio():
    """Return raw byte streams for stdin and the *original* stdout (FD 1).

    Critical: libero / robosuite call ``print()`` during import and env reset
    (e.g. ``[info] using task orders [...]``). Those prints go to FD 1 and
    would corrupt the framed pickle wire. So we (a) reopen FD 1 as a raw
    binary stream we control, and (b) redirect Python's ``sys.stdout`` to
    ``sys.stderr`` so any user-level ``print`` lands in the per-worker log
    file instead of mid-frame.
    """
    raw_stdin = sys.stdin.buffer
    raw_stdout = os.fdopen(1, "wb", buffering=0)
    sys.stdout = sys.stderr  # any subsequent print() goes to FD 2
    return raw_stdin, raw_stdout


def _build_env(init_args):
    """Instantiate one LIBERO env from the init handshake."""
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite_name = init_args["task_suite"]
    task_id = int(init_args["task_id"])

    bench = benchmark.get_benchmark_dict()[suite_name]()
    task = bench.get_task(task_id)
    bddl_root = get_libero_path("bddl_files")
    bddl_file = os.path.join(bddl_root, task.problem_folder, task.bddl_file)
    init_states = bench.get_task_init_states(task_id)

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=int(init_args.get("camera_heights", 256)),
        camera_widths=int(init_args.get("camera_widths", 256)),
        camera_names=list(init_args.get("camera_names", ["agentview", "robot0_eye_in_hand"])),
        controller=str(init_args.get("controller", "OSC_POSE")),
        horizon=int(init_args.get("horizon", 600)),
    )
    return env, init_states, dict(suite_name=suite_name, task_id=task_id, bddl_file=bddl_file)


def main():
    stdin, stdout = _stdio()

    try:
        init_args = read_frame(stdin)
        env, init_states, meta = _build_env(init_args)
        write_frame(stdout, {"ok": True, "meta": meta, "n_init_states": len(init_states)})
    except Exception:
        write_frame(stdout, {"ok": False, "error": traceback.format_exc()})
        return

    while True:
        try:
            msg = read_frame(stdin)
        except EOFError:
            break
        cmd = msg["cmd"]
        try:
            if cmd == "reset":
                raw = env.reset()
                write_frame(stdout, {"obs": adapt_obs(raw)})
            elif cmd == "set_init_state":
                idx = int(msg["idx"]) % len(init_states)
                raw = env.set_init_state(init_states[idx])
                # set_init_state in libero returns either the new obs or None;
                # if None, do an env step with zero action to fetch one.
                if raw is None:
                    raw, _, _, _ = env.step(np.zeros(env.action_dim if hasattr(env, "action_dim") else 7))
                write_frame(stdout, {"obs": adapt_obs(raw)})
            elif cmd == "step":
                action = np.asarray(msg["action"], dtype=np.float64)
                raw, reward, done, info = env.step(action)
                success = bool(env.check_success())
                write_frame(
                    stdout,
                    {
                        "obs": adapt_obs(raw),
                        "reward": float(reward),
                        "done": bool(done),
                        "success": success,
                    },
                )
            elif cmd == "close":
                try:
                    env.close()
                except Exception:
                    pass
                write_frame(stdout, {"ok": True})
                break
            else:
                write_frame(stdout, {"error": f"unknown cmd {cmd!r}"})
        except Exception:
            write_frame(stdout, {"error": traceback.format_exc()})


if __name__ == "__main__":
    main()
