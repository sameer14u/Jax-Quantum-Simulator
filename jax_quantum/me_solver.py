# Lindblad Master Equation Solver (MESolver)
# Implementation using the Vectorization (Superoperator) formalism to convert the Master Equation into a linear matrix equation

import jax.numpy as jnp

class MESolver:
    """Lindblad Master Equation Solver for mixed states."""
    
    @staticmethod
    def build_liouvillian(H, c_ops):
        """
        Constructs the Liouvillian Superoperator Matrix L.
        L = -i(I kron H - H.T kron I) + Sum(Dissipator_k)
        """
        dim = H.shape[0]
        I = jnp.eye(dim, dtype=jnp.complex128)
        
        L_H = -1j * (jnp.kron(I, H) - jnp.kron(H.T, I))
        L_diss = jnp.zeros((dim**2, dim**2), dtype=jnp.complex128)
        
        for C in c_ops:
            Cd = C.conj().T
            CdC = jnp.dot(Cd, C)
            
            term1 = jnp.kron(C.conj(), C) 
            term2 = -0.5 * jnp.kron(I, CdC)
            term3 = -0.5 * jnp.kron(CdC.T, I)
            
            L_diss += (term1 + term2 + term3)
            
        return L_H + L_diss

    @staticmethod
    def drho_dt_vectorized(t, rho_vec, L):
        """Linear ODE for vectorized density matrix."""
        return jnp.dot(L, rho_vec)
