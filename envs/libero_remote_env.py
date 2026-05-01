"""Client wrapper around a libero subprocess running in the libero conda env.

Lives in the ldp env. Spawns one libero subprocess per env instance and proxies
``reset`` / ``set_init_state`` / ``step`` / ``check_success`` / ``close`` calls
over length-prefixed pickle frames.

Stderr is drained by a dedicated thread to a log file (one per worker) so it
never corrupts the stdout pickle stream.
"""
from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Allow running this file directly as a script (`python envs/libero_remote_env.py
# --smoke`) by ensuring the repo root is on sys.path before the relative import.
import sys as _sys
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

from envs.libero_proto import read_frame, write_frame


_LIBERO_PYTHON_CANDIDATES = (
    Path.home() / "anaconda3" / "envs" / "libero" / "bin" / "python",
    Path.home() / "miniconda3" / "envs" / "libero" / "bin" / "python",
    Path("/home/anaconda3/envs/libero/bin/python"),
)
_LIBERO_REPO_CANDIDATES = (
    Path.home() / "Documents" / "LIBERO",
    Path("/data/LIBERO"),
)


def _resolve_first_path(candidates, kind: str, executable: bool = False) -> str:
    for path in candidates:
        if executable:
            if path.is_file() and os.access(path, os.X_OK):
                return str(path)
        elif path.is_dir():
            return str(path)
    env_name = "LIBERO_PYTHON" if executable else "LIBERO_REPO"
    checked = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find {kind}. Set {env_name}; checked: {checked}")


def _drain_to_log(stream, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as f:
        for line in iter(stream.readline, b""):
            f.write(line)
    try:
        stream.close()
    except Exception:
        pass


class LiberoRemoteEnv:
    """Surface used by run_libero_eval workers.

    Args mirror the env_kwargs block in data/cfg/libero_long/img.yaml.
    """

    def __init__(
        self,
        task_suite: str,
        task_id: int,
        camera_heights: int = 256,
        camera_widths: int = 256,
        camera_names=("agentview", "robot0_eye_in_hand"),
        controller: str = "OSC_POSE",
        horizon: int = 600,
        libero_python: Optional[str] = None,
        libero_repo: Optional[str] = None,
        log_dir: Optional[str] = None,
        worker_id: Optional[int] = None,
        # legacy / forwarded kwargs from data.env_params['env_kwargs'] are
        # accepted but ignored (e.g. lowdim_obs/rgb_obs which the env doesn't
        # need at construction time).
        **_unused,
    ):
        self.task_suite = task_suite
        self.task_id = int(task_id)
        self.camera_heights = camera_heights
        self.camera_widths = camera_widths
        self.camera_names = list(camera_names)
        self.controller = controller
        self.horizon = horizon

        py = libero_python or os.environ.get("LIBERO_PYTHON") or _resolve_first_path(
            _LIBERO_PYTHON_CANDIDATES, "LIBERO python", executable=True
        )
        repo = libero_repo or os.environ.get("LIBERO_REPO") or _resolve_first_path(
            _LIBERO_REPO_CANDIDATES, "LIBERO repo"
        )
        log_dir = Path(log_dir or os.environ.get("LIBERO_LOG_DIR", "/tmp/libero_remote_env_logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        wid = worker_id if worker_id is not None else os.getpid()
        self._log_path = log_dir / f"worker_{wid}_task{task_id}.log"

        env = os.environ.copy()
        # Make libero importable from the libero conda env. The libero repo
        # ships its package under repo/libero so PYTHONPATH=repo gives `import
        # libero` access. Also include the LDP repo root so the server can do
        # ``from envs.libero_obs_adapter import ...``.
        env["PYTHONPATH"] = os.pathsep.join([str(_REPO_ROOT), str(repo), env.get("PYTHONPATH", "")])
        env.setdefault("MUJOCO_GL", "egl")
        env.setdefault("PYOPENGL_PLATFORM", "egl")
        env.setdefault("ROBOSUITE_LOG_PATH", str(log_dir / f"robosuite_worker_{wid}.log"))

        self._proc = subprocess.Popen(
            [py, "-u", str(_REPO_ROOT / "envs" / "libero_eval_server.py")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            env=env,
        )
        self._stderr_thread = threading.Thread(
            target=_drain_to_log,
            args=(self._proc.stderr, self._log_path),
            daemon=True,
        )
        self._stderr_thread.start()

        # init handshake
        write_frame(
            self._proc.stdin,
            {
                "task_suite": self.task_suite,
                "task_id": self.task_id,
                "camera_heights": self.camera_heights,
                "camera_widths": self.camera_widths,
                "camera_names": self.camera_names,
                "controller": self.controller,
                "horizon": self.horizon,
            },
        )
        ack = self._read_response()
        if not ack.get("ok"):
            raise RuntimeError(
                f"libero subprocess init failed (task={task_id}): "
                f"{ack.get('error', ack)} - see {self._log_path}"
            )
        self._n_init_states = int(ack.get("n_init_states", 0))
        self._meta = ack.get("meta", {})

    # --- low-level wire ----------------------------------------------------

    def _read_response(self):
        try:
            resp = read_frame(self._proc.stdout)
        except EOFError as e:
            tail = ""
            try:
                if self._log_path.exists():
                    tail = self._log_path.read_text()[-2000:]
            except Exception:
                pass
            raise RuntimeError(
                f"libero subprocess died before responding. stderr tail:\n{tail}"
            ) from e
        if "error" in resp:
            raise RuntimeError(f"libero subprocess error: {resp['error']}")
        return resp

    # --- high-level surface -----------------------------------------------

    @property
    def n_init_states(self):
        return self._n_init_states

    def reset(self):
        write_frame(self._proc.stdin, {"cmd": "reset"})
        return self._read_response()["obs"]

    def set_init_state(self, idx: int):
        write_frame(self._proc.stdin, {"cmd": "set_init_state", "idx": int(idx)})
        return self._read_response()["obs"]

    def step(self, action: np.ndarray):
        write_frame(self._proc.stdin, {"cmd": "step", "action": np.asarray(action)})
        resp = self._read_response()
        return resp["obs"], resp["reward"], resp["done"], {"success": resp["success"]}

    def is_success(self):
        # exposed for parity with rm_env_utils.EvalProc.start which expects a dict
        # NB: requires a prior step; mirroring robosuite's bool-dict shape.
        return {"task": False}  # callers should derive from step's info["success"]

    def close(self):
        if self._proc.poll() is not None:
            return
        try:
            write_frame(self._proc.stdin, {"cmd": "close"})
            try:
                self._read_response()
            except Exception:
                pass
        except Exception:
            pass
        try:
            self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        try:
            self._proc.stdout.close()
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def smoke_test():
    """Run with `python envs/libero_remote_env.py --smoke`."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    env = LiberoRemoteEnv(task_suite="libero_10", task_id=args.task_id)
    print(f"init ok, n_init_states={env.n_init_states}, meta={env._meta}")
    obs = env.set_init_state(0)
    print("after set_init_state, obs keys:", sorted(obs.keys()))
    for k, v in obs.items():
        print(f"  {k}: shape={getattr(v, 'shape', '?')} dtype={getattr(v, 'dtype', '?')}")
    t0 = time.time()
    for i in range(args.steps):
        obs, r, done, info = env.step(np.zeros(7))
        print(f"step {i}: r={r:.3f} done={done} success={info['success']}")
        if done:
            break
    dt = (time.time() - t0) / max(1, args.steps)
    print(f"avg step latency: {dt * 1000:.1f} ms")
    env.close()


if __name__ == "__main__":
    import sys

    if "--smoke" in sys.argv:
        sys.argv.remove("--smoke")
        smoke_test()
    else:
        print("usage: python envs/libero_remote_env.py --smoke")
