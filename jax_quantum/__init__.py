"""
JAX Quantum Simulator Package

A high-performance, GPU-accelerated quantum simulation library built on JAX[cite: 5].
This module exposes the core state preparation, operator generation, 
and dynamical solvers for pure and open quantum systems[cite: 6, 7].
"""

# Solves the time-dependent and time-independent Schrödinger Equation for pure state evolution.
from .se_solver import SESolver

# Solves the Lindblad Master Equation for mixed states using a vectorized superoperator formalism.
from .me_solver import MESolver

# Implements the Monte Carlo Wavefunction (Quantum Jump) method to simulate open systems via pure state trajectories.
from .mcwf_solver import MCWFSolver

# Provides Stochastic Differential Equation solvers, including the Stochastic Master Equation (SME) for continuous measurement.
from .sde_solver import SDESolver

# Generates standard quantum states in a truncated Hilbert space, including Fock, coherent, and thermal states.
from .states import StateFactory

# Constructs essential matrix representations of quantum operators like annihilation/creation and Pauli spin observables.
from .operators import Operators

# Helper function to construct composite system operators (e.g., qubit + cavity) using tensor (Kronecker) products.
from .composite import tensor_product

__all__ = [
    "SESolver",
    "MESolver",
    "MCWFSolver",
    "SDESolver",
    "StateFactory",
    "Operators",
    "tensor_product"
]
