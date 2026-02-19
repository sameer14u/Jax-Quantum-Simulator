def tensor_product(op_A, op_B):
    """
    Constructs composite operators using Kronecker product.
    Equivalent to jax.numpy.kron [cite: 25]
    """
    return jnp.kron(op_A, op_B)

# Example Usage based on document:
# "System composed of subsystems A and B... N_A x N_B" [cite: 24]
# psi_composite = tensor_product(psi_A, psi_B)
