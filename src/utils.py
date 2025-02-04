import numpy as np
from jax.tree_util import tree_map
from torch.utils import data
from torchvision.datasets import MNIST
from jax import numpy as jnp
import orbax.checkpoint as ocp
from flax.training.train_state import TrainState
from flax import nnx
from typing import Any
import os

# https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html#data-loading-with-pytorch
def numpy_collate(batch):
    return tree_map(np.asarray, data.default_collate(batch))

class FlattenAndCast(object):
    def __call__(self, pic):
        return np.ravel(np.array(pic, dtype=jnp.float32))