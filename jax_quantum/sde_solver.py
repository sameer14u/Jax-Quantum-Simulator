# Stochastic Differential Equations (SDE)
# Definitions for the drift and diffusion terms for Stochastic Schrödinger Equation (SSE) and Stochastic Master Equation (SME).
import jax.numpy as jnp

class SDESolver:
    """Stochastic Differential Equation Solvers (SSE/SME)."""
    
    @staticmethod
    def sme_dynamics(t, rho, H, C, eta):
        """
        Returns drift and diffusion for Stochastic Master Equation.
        """
        comm = -1j * (jnp.dot(H, rho) - jnp.dot(rho, H))
        
        Cd = C.conj().T
        CdC = jnp.dot(Cd, C)
        diss = jnp.dot(C, jnp.dot(rho, Cd)) - 0.5 * jnp.dot(CdC, rho) - 0.5 * jnp.dot(rho, CdC)
        
        drift = comm + diss
        
        term1 = jnp.dot(C, rho) + jnp.dot(rho, Cd)
        expect = jnp.trace(term1)
        diffusion = jnp.sqrt(eta) * (term1 - expect * rho)
        
        return drift, diffusion
