import matplotlib.pyplot as plt
import jax, jax.numpy as jnp, optax, jax.random as jr
from src.model import create_model
from src.sampler import Sampler, algorithms
import torch
from functools import partial
from flax.training import train_state
from typing import Any, Dict
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from jax.sharding import PartitionSpec as P, NamedSharding
import os
import random
import orbax.checkpoint as ocp
from etils import epath
import json
from omegaconf import OmegaConf, open_dict
from grain.python import IndexSampler, ShardByJaxProcess, Batch

# https://yilundu.github.io/thesis.pdf
class Trainer:
    def __init__(self, model, params, logger, key):
        self.model = model
        self.hparams = params
        self.key = key
        self.logger = logger # event logger
        dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(f"{self.hparams.log_dir}/{dt}") #log to tensorboard
        self.hparams.chkpt_dir = f"{self.hparams.chkpt_dir}/{dt}"
        self._init_train_state()
        self._configure_sampler()
        self._configure_checkpointer()

    def _init_train_state(self):
        self.n_devices = jax.local_device_count()
        self.logger.info(f"Number of devices found: {self.n_devices}")
        mesh = jax.make_mesh((self.n_devices,), ('batch',))
        model_sharding = NamedSharding(mesh, P())
        self.key, model_key = jr.split(self.key)
        variables = self.model.init(model_key, jnp.ones((1,) + self.hparams.shape))
        self._configure_optimizers(variables)
        model_state = train_state.TrainState.create(
            apply_fn=self.model.apply, params=variables["params"], tx=self.optim
        )
        self.model_state = jax.device_put(model_state, model_sharding)
        self.logger.info("Train state initialized")

    def _configure_optimizers(self, variables):

        scheduler = optax.exponential_decay(
                        init_value=self.hparams.lr,
                        transition_steps=1000,
                        decay_rate=0.97
                    )
        self.optim = optax.chain(
            optax.clip_by_global_norm(1.0),  # Clip by the gradient by the global norm.
            optax.scale_by_adam(),  # Use the updates from adam.
            optax.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler.
            # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
            optax.scale(-1.0)
        )
        self.opt_state = self.optim.init(variables)
        self.logger.info("Optimizers configured")

    def _configure_sampler(self):
        self.key, sampler_key = jr.split(self.key)
        self.sampler = Sampler(
            self.model_state,
            self.hparams.shape,
            sampler_key,
            self.hparams.buffer_size,
            self.hparams.batch_size,
        )
        self.logger.info("Sampler configured")

    def _configure_checkpointer(self):
        options = ocp.CheckpointManagerOptions(max_to_keep=self.hparams.max_epochs, save_interval_steps=1, enable_async_checkpointing=True, create=True)
        path = epath.Path(os.path.abspath(self.hparams.chkpt_dir))
        self.checkpoint_mngr = ocp.CheckpointManager(path, options=options, item_names =('state', 'hparams'))
        self.logger.info("Checkpointer configured")

    def predict(self, x):
        return self.model_state.apply_fn({"params": self.model_state.params}, x)
    
    def fit(self, train_loader: Any, val_loader: Any):
        # To shuffle the data, use a sampler:
        best_model = (float('inf'), 0)

        mesh = jax.make_mesh((self.n_devices,), ('batch',))
        data_sharding = NamedSharding(mesh, P('batch'))

        for epoch, epoch_key in tqdm(
            enumerate(jax.random.split(self.key, self.hparams.max_epochs)),
            position=0,
            total=self.hparams.max_epochs,
        ):
            running_loss = 0.0
            train_pbar =  tqdm(zip(enumerate(train_loader), jax.random.split(epoch_key, len(train_loader))), leave=False, position=1, total=len(train_loader))
            for (i, (batch_x, _)), _key in train_pbar:
                z_plus = batch_x.reshape(-1, *self.hparams.shape)
                z_plus += jnp.clip((jr.normal(_key, z_plus.shape) * 0.005), -1, 1)  # corrupt the original images with some random noise
                z_minus = self.sampler.generate(self.hparams.step_size, self.hparams.steps)
                z = jnp.vstack([z_plus, z_minus])
                z = jax.device_put(z, data_sharding)
                (loss, reg_loss), self.model_state = Trainer._step(
                    self.model_state, z, self.hparams.alpha
                )
                running_loss += loss
                train_pbar.set_description(f"train_loss: {loss}")
                self.writer.add_scalar(f"train_loss/epoch_{epoch}", np.asarray(loss), i)
                self.writer.add_scalar(f"reg_loss/epoch_{epoch}", np.asarray(reg_loss), i)

            self.writer.add_scalar(
                f"train_avg_loss/epoch",
                np.asarray(jax.device_get(running_loss) / len(train_loader)),
                epoch,
            )

            # evaluation loop
            running_loss = 0.0
            val_pbar = tqdm(zip(enumerate(val_loader), jax.random.split(epoch_key, len(val_loader))), leave=False, position=1, total=len(val_loader))
            for (i, (batch_x, _)), _key in val_pbar:
                z_plus = batch_x.reshape(-1, *self.hparams.shape)
                z_plus += jnp.clip((jr.normal(_key, z_plus.shape)), -1, 1)
                z_minus = jr.normal(_key, z_plus.shape)
                z = jnp.vstack([z_plus, z_minus])
                z = jax.device_put(z, data_sharding)
                loss, reg_loss =  Trainer.compute_loss(self.model_state.params, self.model_state, z, self.hparams.alpha, is_train=False)
                running_loss += loss
                val_pbar.set_description(f"val_loss: {loss}")
                self.writer.add_scalar(f"val_loss/epoch_{epoch}/", np.asarray(loss), i)
                self.writer.add_scalar(f"reg_loss/epoch_{epoch}/", np.asarray(reg_loss), i)
   
            self.checkpoint_mngr.save(
                epoch, 
                args=ocp.args.Composite(
                    state = ocp.args.StandardSave(self.model_state),
                    hparams = ocp.args.JsonSave(json.dumps((OmegaConf.to_container(self.hparams, resolve=True)), indent=4))
                    ) 
                )

            # track best model for checkpointing
            best_model = min(best_model, (jax.device_get(running_loss) / len(val_loader), epoch))
            # logger.add_hparams(self.hparams, metric_dict, run_name=self.hparams.run_name)
            
            if epoch>0 and (epoch%self.hparams.log_interval) == 0:
                samples = Trainer.generate(self.model_state, self.hparams.batch_size, self.hparams.shape)
                self.writer.add_images(f"generated_samples/epoch_{epoch}", np.asarray(samples), dataformats="NHWC")
                self.logger.info(f"samples logged to tensorboard {epoch}")

        try:
            self.checkpoint_mngr.wait_until_finished()
            if self.hparams.save_best:
                self.checkpoint_mngr.should_save(best_model[1]) #save only the best model
                self.logger.info(f"best model saved at epoch {best_model[1]} with loss {best_model[0]}")
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")
            
    @staticmethod
    @jax.jit
    def _step(model_state, z, alpha):
        # Executes a training loop.
        (loss, reg_loss), grads = jax.value_and_grad(Trainer.compute_loss, has_aux=True)(
            model_state.params, model_state, z, alpha, is_train=True
        )
        model_state = model_state.apply_gradients(grads=grads)
        return (loss, reg_loss), model_state

    @staticmethod
    @jax.jit
    def compute_loss(params, model_state, z, alpha, is_train=True):
        logits = model_state.apply_fn({"params": params}, z)
        z_plus, z_minus = jnp.split(logits, 2)
        loss = z_minus.mean() - z_plus.mean()
        reg_loss = alpha * (z_plus**2 + z_minus**2).mean() #for regularization
        return jax.lax.cond(is_train, lambda _: loss+reg_loss, lambda _: loss, reg_loss), reg_loss
        
    @staticmethod
    def generate(model_state, bs, img_shape, step_size=256, steps=10):
        samples = jr.normal(jax.random.key(0), (bs, *img_shape))
        out = Sampler.sample(samples, model_state, step_size, steps)
        return out
    
    '''
    @brief: load trained model and saved params from checkpoint and return an instance of the Trainer class
    '''
    @staticmethod
    def load_from_checkpoint(path, load_epoch: int = 0):
        pass