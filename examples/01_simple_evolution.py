import jax.numpy as jnp

# Import directly from the package you just created!
from jax_quantum import StateFactory, Operators, SESolver

def main():
    print(" Initializing JAX Quantum Simulation...")
    
    # 1. Define the system parameters
    N = 10          # Truncated Hilbert space dimension
    omega = 1.0     # Oscillator frequency

    # 2. Create operators using your Operators class
    a = Operators.destroy(N)
    adag = Operators.create(N)
    
    # Define Hamiltonian: H = omega * a^dag a
    H = omega * jnp.dot(adag, a) 

    # 3. Create the initial state: A coherent state with alpha=1.5
    psi0 = StateFactory.coherent_state(N, alpha=1.5)
    print(f"Initial state created. Norm: {jnp.linalg.norm(psi0):.4f}")

    # 4. Perform Time Evolution (Schrödinger Equation)
    t = 2.0  # Time to evolve
    psi_t = SESolver.exact_evolution(psi0, H, t)
    print(f"Evolved to time t={t}. Final state norm: {jnp.linalg.norm(psi_t):.4f}")
    
    # Calculate expected photon number <n> = <psi| a^dag a |psi>
    n_op = Operators.number(N)
    expected_n = jnp.real(jnp.vdot(psi_t, jnp.dot(n_op, psi_t)))
    print(f"Expected photon number <n>: {expected_n:.4f}")
    
    print(" Simulation successful!")

if __name__ == "__main__":
    main()
