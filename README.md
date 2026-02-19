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



## 🎓 Acknowledgments
This is an ongoing research project being developed under the guidance of **Dr. [cite_start]Sangkha Borah** at the **Indian Institute of Technology (IIT) Hyderabad**.

## ⚙️ Installation

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/sameer14u/Jax-Quantum-Simulator.git](https://github.com/sameer14u/Jax-Quantum-Simulator.git)
cd Jax-Quantum-Simulator
pip install -r requirements.txt
