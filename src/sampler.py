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


def ldmc(
    x: jax.Array, model_state: nnx.Module, step_size: int, steps: int
) -> jax.Array:
    key = jax.random.key(0)
    # iterate over steps, adding noise (drawn from a guassian dist) to x
    for sub_key in jax.random.split(key, steps):
        # grad(E(x_k-1))
        jnp.clip(x.__add__(jax.random.normal(sub_key, x.shape)), -1, 1)
        dx = jax.grad(
            lambda x_, p: -model_state.apply_fn({"params": p}, x_).sum()
        )(x, model_state.params)
        jnp.clip(dx, -0.03, 0.03)  # clip gradients
        x += -step_size * dx
        # jnp.clip(x, -1, 1)

    return x


def hmc() -> jnp.ndarray:
    return []


class Sampler:

    def __init__(
        self,
        model_state: nnx.Module,
        shape: tuple,
        key: jnp.ndarray,
        buffer_size: int = 32,
        sample_size: int = 16,
    ):
        self.model_state = model_state  # model to be used as the energy function
        self.sample_size = sample_size  # batch size
        self.shape = shape  # shape of the input data
        self.buffer_size = buffer_size  # max size of the buffer
        self.key, b_key = jax.random.split(key)
        self.buffer = [
            jax.random.uniform(_key, (1,) + self.shape)
            for _key in jax.random.split(b_key, buffer_size)
        ]  # replay buffer to store samples shape (buffer_size, *shape)

    def generate(
        self, step_size: int, steps: int, method: algorithms = algorithms.LDMC
    ):
        self.key, subkey = jax.random.split(self.key)
        if self.buffer:
            # 95% from buffer
            from_buffer = jnp.vstack(
                random.choices(self.buffer, k=int(0.95 * self.sample_size))
            )
            from_normal = jax.random.uniform(
                subkey, (self.sample_size - from_buffer.shape[0],) + self.shape
            )
            z = jnp.vstack([from_buffer, from_normal])
        else:
            # noise from uniform dist
            z = jax.random.uniform(subkey, (self.sample_size,) + self.shape)

        sample = self.sample(
            z, self.model_state, step_size, steps, method
        )  # sample_size * *shape
        self.buffer += jnp.split(sample, self.sample_size)
        # random.choices(self.buffer, k=self.buffer_size)
        self.buffer = self.buffer[: self.buffer_size]

        return sample

    @staticmethod
    def sample(
        x: jnp.ndarray,
        model_state: nnx.Module,
        step_size: int,
        steps: int,
        method: algorithms = algorithms.LDMC,
    ):
        if method == algorithms.LDMC:
            return ldmc(x, model_state, step_size, steps)
        elif method == algorithms.HMC:
            return hmc()
