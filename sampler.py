from flax import nnx
from enum import Enum
import jax
from jax import numpy as jnp
from typing import Dict
import random
# @TODO add more algorithms

algorithms = Enum("algorithm", [("LD", 1), ("HMC", 2)])

"""
    @brief: Langevin Dynamics MCMC
    @param inputs: jax.Array - input data  
    @param model: nnx.Module - model to be used as the energy function
    @param params: Dict - model parameters
    @param step_size: int - step size for langevin dynamics
    @param steps: int - number of steps to take in langevin dynamics
    @return: jax.Array - sampled data
"""
def ld(inputs: jax.Array, model: nnx.Module, params: Dict, step_size: int, steps: int) -> jax.Array:
    key = jax.random.key(0)
    # iterate over steps, adding noise (drawn from a normal dist) to inputs
    # pertubed = [inputs]
    for _ in range(steps):
        # inputs.__add__(noise).clip(-1, 1)
        # grad(E(x_k-1))
        # primals, f_vjp = jax.vjp(lambda x, p: -model.apply(p, x), inputs, params)
        dx = jax.grad(lambda x, p: -model.apply(p, x).sum(), argnums=0)(inputs, params)
        # dx, _ = f_vjp(jnp.ones_like(primals, dtype=jnp.float32))
        key, subkey = jax.random.split(key)
        print(dx)
        # x_k-1 - [step_size * grad(E(x_k-1))] + noise (std=0.05)
        dx = dx.clip(-0.03, 0.03) # clip gradients
        inputs = inputs + -step_size * dx + 0.05*jax.random.normal(subkey, inputs.shape)
        inputs = inputs.clip(-1, 1) # for stability

    return inputs


def hmc() -> jax.Array:
    return []

class Sampler:
    buffer=[]
    sample_size: int

    @staticmethod
    def sample_(
        inputs: jax.Array,
        model: nnx.Module,
        params: Dict,
        method: algorithms,
        step_size: int,
        steps: int,
    ):
        # set in eval mode
        # model.eval()
        # sample = None
        if method == algorithms.LD:
            return ld(inputs, model, params, step_size, steps)
        elif method == algorithms.HMC:
            return hmc()