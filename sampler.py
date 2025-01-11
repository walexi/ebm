from flax import nnx
from enum import Enum
import jax
from jax import numpy as jnp
from typing import Dict
import random
# @TODO add more algorithms

algorithms = Enum("algorithm", [("LDMC", 1), ("HMC", 2)])

"""
    @brief: Langevin Dynamics MCMC
    @param inputs: jax.Array - input data  
    @param model: nnx.Module - model to be used as the energy function
    @param params: Dict - model parameters
    @param step_size: int - step size for langevin dynamics
    @param steps: int - number of steps to take in langevin dynamics
    @return: jax.Array - sampled data
"""
def ldmc(inputs: jax.Array, model: nnx.Module, params: Dict, step_size: int, steps: int) -> jax.Array:
    key = jax.random.key(0)
    # iterate over steps, adding noise (drawn from a normal dist) to inputs
    for _ in range(steps):
        # inputs.__add__(noise).clip(-1, 1)
        # grad(E(x_k-1))
        dx = jax.grad(lambda x, p: -model.apply(p, x).sum(), argnums=0)(inputs, params)
        key, subkey = jax.random.split(key)
        jnp.clip(dx, -0.03, 0.03) # clip gradients
        # x_k-1 - [step_size * grad(E(x_k-1))] + noise (std=0.05)
        inputs+=(-step_size * dx + 0.05*jax.random.normal(subkey, inputs.shape))
        jnp.clip(inputs, -1, 1)

    return inputs

def hmc() -> jax.Array:
    return []

class Sampler:
    buffer=[]
    sample_size: int

    @staticmethod
    def sample_(inputs: jax.Array, model: nnx.Module, params: Dict, method: algorithms, step_size: int, steps: int):
        if method == algorithms.LDMC:
            return ldmc(inputs, model, params, step_size, steps)
        elif method == algorithms.HMC:
            return hmc()