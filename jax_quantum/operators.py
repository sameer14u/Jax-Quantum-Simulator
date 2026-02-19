class Operators:
    # --- Bosonic Operators [cite: 50] ---
    @staticmethod
    def destroy(N):
        """
        Annihilation operator 'a' for truncated size N.
        Upper-diagonal matrix with sqrt(n) elements. [cite: 51, 52]
        """
        # Elements defined by sqrt(n) on the first upper diagonal [cite: 53]
        diagonals = jnp.sqrt(jnp.arange(1, N, dtype=jnp.complex128))
        return jnp.diag(diagonals, k=1)

    @staticmethod
    def create(N):
        """Creation operator a^dag = (a)^T [cite: 54]"""
        return Operators.destroy(N).conj().T

    @staticmethod
    def number(N):
        """Number operator n = a^dag * a [cite: 55]"""
        a = Operators.destroy(N)
        adag = Operators.create(N)
        return jnp.dot(adag, a)

    @staticmethod
    def position(N):
        """Position operator x = (a + a^dag) / sqrt(2) [cite: 59]"""
        a = Operators.destroy(N)
        adag = Operators.create(N)
        return (a + adag) / jnp.sqrt(2)

    @staticmethod
    def momentum(N):
        """Momentum operator p = (a - a^dag) / (i * sqrt(2)) [cite: 59]"""
        a = Operators.destroy(N)
        adag = Operators.create(N)
        return (a - adag) / (1j * jnp.sqrt(2))

    # --- Spin/Qubit Operators [cite: 58] ---
    @staticmethod
    def sigma_x():
        """Pauli X [cite: 62]"""
        return jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)

    @staticmethod
    def sigma_y():
        """Pauli Y [cite: 62]"""
        return jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)

    @staticmethod
    def sigma_z():
        """Pauli Z [cite: 62]"""
        return jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)
    
    @staticmethod
    def sigma_plus():
        """Raising operator sigma_+ [cite: 63]"""
        sx = Operators.sigma_x()
        sy = Operators.sigma_y()
        return 0.5 * (sx + 1j * sy)

    @staticmethod
    def sigma_minus():
        """Lowering operator sigma_- [cite: 63]"""
        sx = Operators.sigma_x()
        sy = Operators.sigma_y()
        return 0.5 * (sx - 1j * sy)
