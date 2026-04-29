"""Bridge between the LIBERO env's runtime obs schema and the LDP dataset
schema.

LIBERO env returns per-step:
    agentview_image                (H, W, 3) uint8
    robot0_eye_in_hand_image       (H, W, 3) uint8
    robot0_eef_pos                 (3,) float
    robot0_eef_quat                (4,) float (xyzw quat)
    robot0_gripper_qpos            (2,) float
    object                         (D,)
    ...

The LDP dataset (preprocessed at /home/dodo/data1/libero/256/) stores instead:
    agentview_rgb, eye_in_hand_rgb, ee_pos, ee_ori (3-d Euler), ee_states
    (=concat(ee_pos, ee_ori), 6-d), gripper_states (2-d), joint_states (7-d)

To keep the planner/IDM seeing the same key namespace at training and
inference, this adapter rewrites every reset/step obs from the LIBERO runtime
schema into the dataset schema. Lives on the *server* side so the wire
protocol carries dataset-keyed obs only.

Quat convention: scipy ``Rotation.from_quat([x, y, z, w]).as_euler('xyz')``.
The dataset's ``ee_ori`` was computed by an upstream preprocessor; if a smoke
test reveals a different convention we will revisit (likely 'XYZ' intrinsic vs
'xyz' extrinsic, or zyx ordering).
"""
from __future__ import annotations

import numpy as np

try:
    from scipy.spatial.transform import Rotation as R
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


_KEY_RENAME = {
    "agentview_image": "agentview_rgb",
    "robot0_eye_in_hand_image": "eye_in_hand_rgb",
}


def _quat_to_euler_xyz(q):
    if _HAS_SCIPY:
        # scipy expects (x, y, z, w); robosuite's robot0_eef_quat IS xyzw
        return R.from_quat(np.asarray(q, dtype=np.float64)).as_euler("xyz")
    # fallback (no scipy in libero env? unlikely)
    x, y, z, w = q
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float64)


def adapt_obs(raw_obs):
    """Rewrite a single env obs dict from runtime keys to dataset keys.

    Output dict keys (only those used by training; extras are dropped):
        agentview_rgb, eye_in_hand_rgb (H, W, 3) uint8
        ee_pos (3,), ee_ori (3,), ee_states (6,), gripper_states (2,),
        joint_states (7,)  - if available, else omitted.
    """
    out = {}
    for src, dst in _KEY_RENAME.items():
        if src in raw_obs:
            arr = raw_obs[src]
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            out[dst] = arr

    if "robot0_eef_pos" in raw_obs:
        ee_pos = np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float64)
        out["ee_pos"] = ee_pos
    if "robot0_eef_quat" in raw_obs:
        ee_ori = _quat_to_euler_xyz(raw_obs["robot0_eef_quat"])
        out["ee_ori"] = ee_ori
    if "ee_pos" in out and "ee_ori" in out:
        out["ee_states"] = np.concatenate([out["ee_pos"], out["ee_ori"]], axis=0)

    if "robot0_gripper_qpos" in raw_obs:
        out["gripper_states"] = np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float64)
    if "robot0_joint_pos" in raw_obs:
        out["joint_states"] = np.asarray(raw_obs["robot0_joint_pos"], dtype=np.float64)
    elif "joint_states" in raw_obs:
        out["joint_states"] = np.asarray(raw_obs["joint_states"], dtype=np.float64)

    return out
