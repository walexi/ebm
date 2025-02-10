from flax import linen as nn


class CNN(nn.Module):
    """A simple CNN model."""

    img_size: int = 32
    out_dim: int = 1

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.img_size, kernel_size=(3, 3))(x)
        x = nn.swish(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.swish(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.swish(x)
        x = nn.Dense(features=self.out_dim)(x)
        return x


def create_model(img_size, out_dim):
    return CNN(img_size=img_size, out_dim=out_dim)
