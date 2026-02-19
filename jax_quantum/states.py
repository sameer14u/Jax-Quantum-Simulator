class StateFactory:
    @staticmethod
    def fock_state(N, n):
        """
        Creates a Fock state |n> in a truncated Hilbert space of size N.
        Represented as a one-hot vector. [cite: 19, 20]
        """
        # "In the computational basis, these are represented as one-hot vectors" [cite: 20]
        state = jnp.zeros((N, 1), dtype=jnp.complex128)
        state = state.at[n, 0].set(1.0)
        return state

    @staticmethod
    def coherent_state(N, alpha):
        """
        Creates a coherent state |alpha> using the displacement operator D(alpha).
        |alpha> = D(alpha)|0> = exp(alpha*a^dag - alpha.conj()*a)|0> [cite: 37]
        """
        # We need the annihilation operator 'a' to construct this (defined below).
        # This function assumes 'a' and 'adag' are available or reconstructs them locally.
        a = jnp.diag(jnp.sqrt(jnp.arange(1, N, dtype=jnp.complex128)), k=1)
        adag = a.conj().T
        
        # Construct displacement operator logic
        # D(alpha) = exp(alpha * a^dag - alpha* * a) [cite: 37]
        exponent = alpha * adag - jnp.conj(alpha) * a
        D = expm(exponent) # "In JAX, this implies matrix exponentiation" [cite: 39]
        
        # Apply to vacuum |0>
        vacuum = StateFactory.fock_state(N, 0)
        return jnp.dot(D, vacuum)

    @staticmethod
    def thermal_state(H, beta):
        """
        Creates a thermal mixed state (density matrix) rho_th.
        rho_th = exp(-beta * H) / Tr(exp(-beta * H)) [cite: 44]
        """
        exp_H = expm(-beta * H)
        partition_func = jnp.trace(exp_H)
        rho_th = exp_H / partition_func
        return rho_th
