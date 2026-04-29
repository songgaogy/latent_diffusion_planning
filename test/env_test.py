# test_torch_jax.py

import sys

print("=" * 50)
print("Python version:", sys.version)
print("=" * 50)

# =======================
# PyTorch Test
# =======================
print("\n[PyTorch Test]")
try:
    import torch

    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("cuda device:", torch.cuda.get_device_name(0))
        print("cudnn version:", torch.backends.cudnn.version())

        # simple tensor op
        x = torch.randn(1024, 1024, device="cuda")
        y = torch.randn(1024, 1024, device="cuda")
        z = x @ y

        print("tensor matmul OK, shape:", z.shape)

    else:
        print("running on CPU")
except Exception as e:
    print("PyTorch FAILED:", repr(e))


# =======================
# JAX Test
# =======================
print("\n[JAX Test]")
try:
    import jax
    import jax.numpy as jnp

    print("jax version:", jax.__version__)
    print("devices:", jax.devices())

    # simple op
    x = jnp.ones((1024, 1024))
    y = jnp.ones((1024, 1024))

    @jax.jit
    def matmul(a, b):
        return a @ b

    z = matmul(x, y)

    print("jax matmul OK, shape:", z.shape)

except Exception as e:
    print("JAX FAILED:", repr(e))


# =======================
# Cross sanity (memory)
# =======================
print("\n[Cross Check]")
try:
    import torch
    import jax
    import numpy as np

    a = torch.randn(4, 4).cpu().numpy()
    b = jax.numpy.array(a)

    print("torch -> numpy -> jax OK")

except Exception as e:
    print("Cross FAILED:", repr(e))

print("\nDONE")