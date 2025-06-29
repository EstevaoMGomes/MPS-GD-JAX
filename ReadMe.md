# MPS-GD-JAX: A JAX-based framework for optimizing Matrix Product States (MPS) using Gradient Descent
This project provides a JAX implementation of the MPS-GD algorithm, which is designed to optimize MPS representations of quantum states.

## Features
- JAX-based implementation for efficient computation and automatic differentiation
- Support for various MPS operations including normalization, contraction, and gradient computation
- Example usage with a simple Hamiltonian and MPS representation

## Installation
To use this project, ensure you have JAX and OPTAX installed. You can install JAX with
```bash
pip install jax
```
and OPTAX with
```bash
pip install optax
```
## Usage
The main functionality is encapsulated in the `optimize.py` module, where:
- Initialize an MPS representation
- Compute the energy of the MPS representation with respect to a given Hamiltonian        
- Compute the gradient of the energy with respect to the MPS parameters
- Perform gradient descent optimization on the MPS

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details

## References
- [JAX Documentation](https://jax.readthedocs.io/en/latest/)
- [OPTAX Documentation](https://optax.readthedocs.io/en/latest/)
- [Matrix Product States](https://en.wikipedia.org/wiki/Matrix_product_state)
- [Gradient Descent Optimization](https://en.wikipedia.org/wiki/Gradient_descent)
