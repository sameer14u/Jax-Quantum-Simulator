# JAX Quantum Simulator

[![JAX](https://img.shields.io/badge/Powered%20by-JAX-blue.svg)](https://github.com/google/jax)
[![Python](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance, GPU-accelerated quantum simulation package built entirely in JAX. This library bridges the gap between quantum optics theory and vector-space operations, making it suitable for automatic differentiation and XLA compilation.

## ✨ Key Features
* **High-Performance**: Utilizes dense linear algebra to fully utilize TPU/GPU tensor cores for small-to-medium quantum systems (N < 100).
* **Comprehensive Solvers**: Includes the Schrödinger equation (SESolver), a vectorized Lindblad Master Equation (MESolver), and Stochastic Differential Equations (SSE/SME).
* **Parallel Trajectories**: Exploit JAX's `vmap` for highly parallel simulation of thousands of Monte Carlo Wavefunction (Quantum Jump) trajectories on a single GPU.
* **Differentiable**: By writing the solver in pure JAX, you can differentiate the final fidelity with respect to Hamiltonian parameters for optimal control tasks like GRAPE or CRAB.

## 🧠 How It Works (Simulation Architecture)



[cite_start]The simulator bridges quantum optics theory with vector-space operations suitable for automatic differentiation[cite: 6]. The general computational workflow is:

1. [cite_start]**Space Truncation:** Infinite-dimensional Hilbert spaces (like bosonic modes) are truncated to a numerical dimension N[cite: 18, 19].
2. [cite_start]**Operator & State Construction:** Quantum states and operators are mapped to dense JAX arrays. [cite_start]For composite systems (e.g., a qubit coupled to a cavity), the simulator constructs the expanded space using Kronecker products[cite: 24, 25].
3. **Dynamics Integration:** The Hamiltonian and initial states are passed to a specific solver:
   * [cite_start]**SESolver:** Integrates the Schrödinger equation directly for pure state evolution[cite: 68, 69].
   * [cite_start]**MESolver:** Vectorizes the density matrix from an (N x N) matrix into an (N^2 x 1) vector[cite: 81]. [cite_start]This maps the Lindblad Master Equation to a linear system governed by the Liouvillian superoperator, solved via highly optimized matrix-vector multiplication[cite: 82, 84, 88].
   * **MCWFSolver:** Evolves a pure state under a non-unitary effective Hamiltonian. [cite_start]When the state norm drops below a dynamically generated random threshold, a quantum jump is applied[cite: 89, 93].

## 🗺️ Roadmap & Future Work
* Integration of time-dependent Hamiltonians for optimal control pulses.
* [cite_start]Full utilization of JAX's `vmap` to parallelize thousands of Monte Carlo trajectories on a single GPU[cite: 93].
* [cite_start]Expansion of the SDESolver for continuous homodyne/heterodyne measurement diffusion (Stochastic Schrödinger Equation)[cite: 96, 97].

## 🎓 Acknowledgments
This is an ongoing research project being developed under the guidance of **Dr. [cite_start]Sangkha Borah** at the **Indian Institute of Technology (IIT) Hyderabad**.

## ⚙️ Installation

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/sameer14u/Jax-Quantum-Simulator.git](https://github.com/sameer14u/Jax-Quantum-Simulator.git)
cd Jax-Quantum-Simulator
pip install -r requirements.txt
