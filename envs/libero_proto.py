"""Length-prefixed pickle framing used by the LIBERO eval client/server bridge.

Frame format:  [ 4-byte big-endian length ][ pickle.dumps(payload) ]

Designed to live in *both* conda envs (no third-party deps) so the libero-env
subprocess can ``import envs.libero_proto`` directly off the repo root.
"""
from __future__ import annotations

import pickle
import struct


_HEADER = struct.Struct(">I")


def write_frame(stream, obj):
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(_HEADER.pack(len(payload)))
    stream.write(payload)
    stream.flush()


def _read_exact(stream, n):
    buf = b""
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise EOFError(f"frame truncated: wanted {n} got {len(buf)}")
        buf += chunk
    return buf


def read_frame(stream):
    header = _read_exact(stream, _HEADER.size)
    (n,) = _HEADER.unpack(header)
    payload = _read_exact(stream, n)
    return pickle.loads(payload)
