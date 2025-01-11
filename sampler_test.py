from absl.testing import absltest, parameterized
import jax
from jax import numpy as jnp
import logging
from model import create_model
from sampler import Sampler, algorithms
from torchvision.datasets import MNIST
import random

jax.config.update("jax_disable_most_optimizations", True)
logging.getLogger('jax._src.xla_bridge').addFilter(lambda _: False)


class SamplerTest(parameterized.TestCase):
    """Test cases for sampler and MCMC algos."""

    def setUp(self):
        """Set up the test."""
        self.sampler = Sampler()
        self.img_size = 28
        self.num_classes = 10
        self.shape = (1, self.img_size, self.img_size, 1)
        self.model = create_model(self.img_size, self.num_classes)
        self.params = self.model.init(jax.random.key(0), jnp.ones(self.shape))
        self.ds = MNIST(root="data", train=True, download=True) # temporary fix, remove later
    
    @parameterized.product(
        method=[algorithms.LD], step_size=[10, 20, 50], steps=[100, 200]
    )
    def test_algos(self, method: algorithms, step_size: int, steps: int):
        """Test implemented sampling techniques."""

        example = jnp.array(random.choice(self.ds)[0], dtype=jnp.float32).reshape(self.shape)
        sample = Sampler.sample_(example, self.model, self.params, method, step_size, steps)
        # @TODO add more tests
        self.assertEqual(sample.shape, example.shape)
    
    

if __name__ == "__main__":
    absltest.main()
