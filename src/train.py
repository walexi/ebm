import matplotlib.pyplot as plt
import jax, jax.numpy as jnp, optax, jax.random as jr
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from src.model import create_model
from src.sampler import Sampler, algorithms
import torch
from functools import partial
from flax.training import train_state
from typing import Any, Dict
from tqdm import tqdm
import numpy as np
# https://yilundu.github.io/thesis.pdf
# @TODO add callbacks to log samples, add validation loop, optimize training loop
class Trainer:
    def __init__(self, model, params, logger, key):
        self.model = model
        print('type', type(params))
        self.hparams = params
        self.key = key
        self.logger = logger
        self._init_train_state()
        self._configure_sampler()

    def _init_train_state(self):
        self.key, model_key = jr.split(self.key)
        variables = self.model.init(model_key, jnp.ones((1,) + self.hparams.shape))
        self._configure_optimizers(variables)
        self.model_state = train_state.TrainState.create(apply_fn=self.model.apply, params=variables['params'], tx=self.optim)

    def _configure_optimizers(self, variables):
        self.optim = optax.adam(self.hparams.lr)
        self.opt_state = self.optim.init(variables)

    def _configure_sampler(self):
        self.key, sampler_key = jr.split(self.key)
        self.sampler = Sampler(
            self.model_state,
            self.hparams.shape,
            sampler_key,
            self.hparams.buffer_size,
            self.hparams.sample_size,
        )

    def fit(self, train_loader: Any, val_loader: Any):
        for epoch, epoch_key in tqdm(enumerate(jax.random.split(self.key, self.hparams.max_epochs)), position=0, total=self.hparams.max_epochs):
            running_loss = 0.0
            for (i, (batch_x, _)), _key in tqdm( zip(enumerate(train_loader), jax.random.split(epoch_key, len(train_loader))), leave=False, position=1, total=len(train_loader)):
                z_plus = batch_x.reshape(-1, self.hparams.image_size, self.hparams.image_size, 1)
                z_plus += (
                    jr.normal(_key, z_plus.shape) * 0.005
                )  # corrupt the original images with some random noise
                z_minus = self.sampler.generate(self.hparams.step_size, self.hparams.steps)
                z = jnp.vstack([z_plus, z_minus])
                loss, self.model_state = Trainer._step(
                    self.model_state,
                    z,
                    self.hparams.alpha
                ) 
                running_loss += loss
                self.logger.add_scalar(f'train_loss/epoch_{epoch}', np.asarray(loss), i)

            self.logger.add_scalar(f'train_avg_loss/epoch_{epoch}', np.asarray(running_loss/len(train_loader)), epoch)

        # logger.add_hparams(self.hparams, metric_dict, run_name=self.hparams.run_name)   

    @staticmethod
    @jax.jit
    def _step(model_state, z, alpha, is_train=True):
        # Executes a training loop.
        loss, grads = jax.value_and_grad(Trainer.compute_loss)(model_state.params, model_state, z, alpha)
        if is_train:
            model_state = model_state.apply_gradients(grads=grads)
        return loss, model_state


    @staticmethod
    @jax.jit
    def compute_loss(params, model_state, z, alpha):
        logits = model_state.apply_fn({"params":params}, z)
        z_plus, z_minus = jnp.split(logits, 2)
        reg_loss = jax.lax.stop_gradient(alpha * (z_plus**2 - z_minus**2).mean())
        cdiv_loss = z_minus.mean() - z_plus.mean()
        loss = reg_loss + cdiv_loss

        return loss