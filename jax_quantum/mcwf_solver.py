# Monte Carlo Wavefunction (MCWF / SSE)
# Logic for the Effective Hamiltonian and Jump condition
import jax
import jax.numpy as jnp

class MCWFSolver:
    """Monte Carlo Wavefunction (Quantum Jump) Solver."""
    
    @staticmethod
    def effective_hamiltonian(H, c_ops):
        """H_eff = H - (i/2) * Sum(C_k^dag C_k)"""
        sum_cdc = jnp.zeros_like(H)
        for C in c_ops:
            sum_cdc += jnp.dot(C.conj().T, C)
            
        H_eff = H - 0.5j * sum_cdc
        return H_eff

    @staticmethod
    def apply_jump(psi, c_ops, key):
        """Applies a quantum jump based on collapse operators."""
        choice_idx = jax.random.randint(key, (1,), 0, len(c_ops))[0]
        C = c_ops[choice_idx]
        
        new_psi = jnp.dot(C, psi)
        norm = jnp.linalg.norm(new_psi)
        return new_psi / norm
