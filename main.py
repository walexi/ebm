from src.train import Trainer
from src.utils import numpy_collate, FlattenAndCast
from src.model import create_model
from torchvision import transforms, datasets
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from datetime import datetime
from torch.utils.data import DataLoader
import numpy as np
import jax
from jax import numpy as jnp
import random
import logging

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_name="config",
    config_path="configs",
)
def main(hparams: DictConfig) -> None:

    train_set = datasets.MNIST(
        root=hparams.dataset_path,
        train=True,
        transform=FlattenAndCast(),
        download=True,
    )
    test_set = datasets.MNIST(
        root=hparams.dataset_path,
        train=False,
        transform=FlattenAndCast(),
        download=True,
    )
    train_dataloader = DataLoader(
        train_set,
        batch_size=hparams.batch_size,
        num_workers=0,
        collate_fn=numpy_collate,
        sampler=np.random.permutation(100), #for debugging
    )
    val_dataloader = DataLoader(
        test_set,
        batch_size=hparams.batch_size,
        num_workers=0,
        collate_fn=numpy_collate,
        sampler=np.random.permutation(100),
    )

    key = jax.random.key(0)
    model = create_model(hparams.image_size, hparams.n_classes)
    
    # logging.basicConfig(filename='myapp.log', level=logging.INFO)
    # logger.info('Started')
    # logger.info('Finished')

    with open_dict(hparams):
        hparams.shape = (hparams.image_size,) * 2 + (hparams.channels,)
        hparams.run_name = datetime.now().strftime("%Y%m%d-%H%M%S")

    print(OmegaConf.to_container(hparams, resolve=True))
    trainer = Trainer(model, hparams, logger, key)
    trainer.fit(train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
