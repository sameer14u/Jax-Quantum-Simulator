import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.scipy.linalg import expm

# 1. Type Precision: Enable x64 for complex128 support
# "Quantum simulations are sensitive to precision." 
jax.config.update("jax_enable_x64", True)

def get_device_info():
    """Helper to check if JAX is using GPU/TPU as intended [cite: 15]"""
    print(f"JAX Backend: {jax.lib.xla_bridge.get_backend().platform}")
    print(f"Devices: {jax.devices()}")
