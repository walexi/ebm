from absl.testing import absltest, parameterized
import jax
from jax import numpy as jnp
import logging
from src.model import create_model
from src.sampler import Sampler, algorithms
from torchvision.datasets import MNIST
import random

jax.config.update("jax_disable_most_optimizations", True)
logging.getLogger("jax._src.xla_bridge").addFilter(lambda _: False)


class SamplerTest(parameterized.TestCase):
    """Test cases for sampler and MCMC algos."""

    def setUp(self):
        """Set up the test."""
        self.img_size = 28
        self.num_classes = 10
        self.shape = (self.img_size, self.img_size, 1)
        self.model = create_model(self.img_size, self.num_classes)
        self.params = self.model.init(jax.random.key(0), jnp.ones((1,) + self.shape))
        self.sampler = Sampler(self.model, self.params, self.shape, jax.random.key(0))

    @parameterized.product(
        method=[algorithms.LDMC], step_size=[10, 20, 50], steps=[40, 100]
    )
    def test_algos(self, method: algorithms, step_size: int, steps: int):
        """Test implemented sampling techniques."""
        example = jax.random.gumbel(jax.random.key(0), (1,) + self.shape)
        sample = Sampler.sample(
            example, self.model, self.params, method, step_size, steps
        )
        # @TODO add more tests
        self.assertEqual(sample.shape, example.shape)

    @parameterized.product(
        method=[algorithms.LDMC],
        step_size=[10, 20, 50],
        steps=[40, 100],
        buffer_size=[40, 80],
        sample_size=[16, 32],
    )
    def test_generate(self, buffer_size, sample_size, step_size, steps, method):
        """Test the generate method."""
        self.sampler.buffer_size = buffer_size
        self.sampler.sample_size = sample_size
        sample = self.sampler.generate(step_size, steps, method)
        self.assertEqual(sample.shape, (sample_size,) + self.shape)
        self.assertBetween(len(self.sampler.buffer), 0, buffer_size)


if __name__ == "__main__":
    absltest.main()
