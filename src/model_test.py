from absl.testing import absltest, parameterized
import jax
from jax import numpy as jnp
import logging
from model import create_model

jax.config.update("jax_disable_most_optimizations", True)
logging.getLogger('jax._src.xla_bridge').addFilter(lambda _: False)

class ModelTest(parameterized.TestCase):
    """Test cases for CNN model definition."""

    @parameterized.product(
        img_size=[28, 32, 64], num_classes=[10, 20], n_channels=[1, 3]
    )
    def test_model(self, img_size, num_classes, n_channels):
        """Tests CNN model definition and output (variables)."""
        model = create_model(img_size, num_classes)
        inputs = jnp.ones((1, img_size, img_size, n_channels))
        variables = model.init(jax.random.key(0), inputs)
        out = model.apply(variables, inputs)

        self.assertLen(variables, 1)
        self.assertLen(variables["params"], 4)
        self.assertEqual(out.shape, (1, num_classes))


if __name__ == "__main__":
    absltest.main()
