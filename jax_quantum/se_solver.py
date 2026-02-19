# chrödinger Equation Solver (SESolver)
# Implementation of the time-dependent Schrödinger equation

class SESolver:
    @staticmethod
    def dpsi_dt(t, psi, H_func, args):
        """
        Computes the time derivative for the ODE solver.
        d|psi>/dt = -i * H(t) * |psi> 
        
        Args:
            t: Time
            psi: Current state vector
            H_func: Function f(t, args) returning the Hamiltonian matrix
            args: Arguments for H_func
        """
        H = H_func(t, *args)
        return -1j * jnp.dot(H, psi)

    @staticmethod
    def exact_evolution(psi0, H, t):
        """
        For time-independent H, solution is exp(-iHt)|psi(0)> [cite: 72]
        """
        U = expm(-1j * H * t)
        return jnp.dot(U, psi0)
