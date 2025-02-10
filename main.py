from src.train import Trainer
from src.utils import numpy_collate, FlattenAndCast
from torchvision import datasets
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from datetime import datetime
from torch.utils.data import  DataLoader
import numpy as np
import jax
from src.model import create_model
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(filename='ebm.log', level=logging.INFO)

@hydra.main(
    version_base=None,
    config_name="config",
    config_path="configs",
)
def main(hparams: DictConfig) -> None:

    train_set = datasets.CIFAR10(
        root=hparams.dataset_path,
        train=True,
        transform=FlattenAndCast(),
        download=True,
    )
    val_set = datasets.CIFAR10(
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
        sampler=np.random.permutation(int(0.5*len(train_set))), #for debugging
    )
    val_dataloader = DataLoader(
        val_set,
        batch_size=hparams.batch_size,
        num_workers=0,
        collate_fn=numpy_collate,
        sampler=np.random.permutation(int(0.5*len(val_set))),
    )
    
    with open_dict(hparams):
        hparams.shape = (hparams.image_size,) * 2 + (hparams.channels,)
        hparams.run_name = datetime.now().strftime("%Y%m%d-%H%M%S")

    print(OmegaConf.to_container(hparams, resolve=True))

    key = jax.random.key(0)
    model = create_model(hparams.image_size, hparams.out_dim)
    trainer = Trainer(model, hparams, logger, key)
    trainer.fit(train_dataloader, val_dataloader)

if __name__ == "__main__":
    main()
