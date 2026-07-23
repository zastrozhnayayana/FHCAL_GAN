import contextlib
from typing import Tuple, Generator, Dict, Any, Optional, ContextManager, Callable
from abc import ABC, abstractmethod

import torch
import torch.utils.data
from torch import optim
from tqdm import tqdm

from pipeline._aux import calc_grad_norm
from pipeline.data import collate_fn, move_batch_to, get_random_infinite_dataloader
from pipeline.device import get_local_device
from pipeline.gan import GAN
from pipeline.logger import GANLogger
from pipeline.metrics import Metric
from pipeline.metrics_logging import log_metric
from pipeline.normalization import update_normalizers_stats
from pipeline.predicates import TrainPredicate
from pipeline.storage import ModelDir


# Обёртка над всем необходимым для шага градиентного спуска (оптимизатор + расписание lr)
class Stepper:
    def __init__(self, optimizer: optim.Optimizer) -> None:
        self.optimizer = optimizer

    def step(self) -> None:
        self.optimizer.step()

    def state_dict(self) -> Dict[str, Any]:
        return {
            'optimizer': self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state_dict['optimizer'])


class GanEpochTrainer(ABC):
    @abstractmethod
    def train_epoch(self, gan_model: GAN,
                    train_dataset: torch.utils.data.Dataset,
                    generator_stepper: Stepper, critic_stepper: Stepper,
                    logger: Optional[GANLogger] = None) -> None:
        pass


def check_tensor(x: torch.Tensor, prefix: str = ''):
    msg = prefix
    if x.isnan().any():
        msg += 'NaNs'
    elif x.isinf().any():
        msg += '+infs'
    elif x.isneginf().any():
        msg += '-infs'
    else:
        return
    raise ValueError(msg)

# распределение энергии по слоям должно быть правдоподобным
def layer_fraction_loss(real_x, fake_x, eps=1e-8):
    real_layer = real_x.sum(dim=(2, 3))  # (batch, 7)
    fake_layer = fake_x.sum(dim=(2, 3))

    real_total = real_layer.sum(dim=1, keepdim=True) # (batch, )
    fake_total = fake_layer.sum(dim=1, keepdim=True)

    real_frac = real_layer.detach() / (real_total.detach() + eps) # (batch, 7) - доля каждого слоя в суммарной энергии (в каждой строке значения суммируются в 1)
    fake_frac = fake_layer / (fake_total + eps)
    # хотим, чтобы fake_frac был ближе к real_frac, поэтому хотим, чтобы градиенты текли через fake_layer и fake_total
    # а real_frac не хотим менять

    return torch.mean((fake_frac - real_frac) ** 2) # MSE

# квантили по суммарной энергии должны быть правдоподобными (хотим, чтобы распределение суммарной энергии было правдоподобным)
def quantile_energy_loss(real_x: torch.Tensor, fake_x: torch.Tensor) -> torch.Tensor:
    """
    Дополнительный loss для генератора: подгоняет квантили распределения полной энергии.
    Градиент идет только через fake_x, real_x используется как target.
    real_x/fake_x ожидаются в формате (batch, layers, rows, cols), например (B, 7, 7, 5).
    """
    real_e = real_x.sum(dim=tuple(range(1, real_x.ndim))).detach() # (B, )
    fake_e = fake_x.sum(dim=tuple(range(1, fake_x.ndim))) # (B, )

    qs = torch.tensor([0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99], device=fake_x.device)

    real_q = torch.quantile(real_e, qs) # (7, B)
    fake_q = torch.quantile(fake_e, qs) # (7, B)

    return torch.mean((fake_q - real_q) ** 2)

class WganEpochTrainer(GanEpochTrainer):
    def __init__(
        self,
        n_critic: int = 5,
        batch_size: int = 64,
        lambda_energy: float = 0.002,
        lambda_sparsity: float = 0.005,
        lambda_layer_fraction: float = 0.1,
        lambda_quantile: float = 0.005,
        debug_every: int = 50,
    ) -> None:
        self.n_critic = n_critic
        self.batch_size = batch_size
        self.lambda_energy = lambda_energy
        self.lambda_sparsity = lambda_sparsity
        self.lambda_layer_fraction = lambda_layer_fraction
        self.lambda_quantile = lambda_quantile
        self.debug_every = debug_every

        self.gen_batch_cnt = 0
        self.disc_batch_cnt = 0
        self.loss_arr = []

    def get_loss_arr(self):
        return self.loss_arr

    def train_epoch(self, gan_model: GAN,
                    train_dataset: torch.utils.data.Dataset,
                    generator_stepper: Stepper, critic_stepper: Stepper,
                    logger: Optional[GANLogger] = None) -> None:
        # даталоадер для генератора (задаёт длину эпохи)
        dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            shuffle=True
        )
        # даталоадер для критика
        random_dataloader = get_random_infinite_dataloader(
            train_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn
        )
        random_dataloader_iter = iter(random_dataloader)

        # берёт батч, в генератор отдаёт condition (gen_batch_y), получает распределение (gen_batch_x)
        def get_batches(real_batch) -> Tuple[torch.Tensor, torch.Tensor, Any, Any]:
            real_batch_x, real_batch_y = move_batch_to(real_batch, get_local_device())
            gen_batch_y = real_batch_y
            noise_batch_z = gan_model.gen_noise(len(real_batch_x)).to(get_local_device())
            gen_batch_x = gan_model.generator(noise_batch_z, gen_batch_y).to(get_local_device())
            return gen_batch_x, real_batch_x, gen_batch_y, real_batch_y

        critic_adv_loss_total = 0.
        critic_loss_total = 0.
        gen_adv_loss_total = 0.
        gen_loss_total = 0.
        gen_energy_loss_total = 0.
        gen_sparsity_loss_total = 0.
        gen_layer_fraction_loss_total = 0.
        gen_quantile_loss_total = 0.
        disc_grad_norm_total = 0.
        gen_grad_norm_total = 0.

        for batch_index, generator_batch in enumerate(tqdm(dataloader)):
            
            # не хотим, чтобы градиенты считались для генератора, т. к. мы будем обновлять сейчас дискриминатор
            gan_model.generator.requires_grad_(False)

            for t in range(self.n_critic):
                self.disc_batch_cnt += 1

                real_batch = next(random_dataloader_iter)
                gen_batch_x, real_batch_x, gen_batch_y, real_batch_y = get_batches(real_batch)

                check_tensor(gen_batch_x, 'Generated values contain ')

                disc_real_vals = gan_model.discriminator(real_batch_x, real_batch_y)
                check_tensor(disc_real_vals, 'Discriminator values for real data contain ')

                disc_gen_vals = gan_model.discriminator(gen_batch_x, gen_batch_y)
                check_tensor(disc_gen_vals, 'Discriminator values for generated data contain ')

                loss = - (disc_real_vals - disc_gen_vals).mean()
                critic_adv_loss_total += loss.item() * len(gen_batch_x)

                loss.backward()

                disc_grad_norm = calc_grad_norm(gan_model.discriminator) # L2-норма
                disc_grad_norm_total += disc_grad_norm

                if logger is not None:
                    logger.log_metrics(
                        data={
                            'train/discriminator/batch_loss': loss.item(),
                            'train/discriminator/batch_grad_norm': disc_grad_norm
                        },
                        period='disc_batch',
                        period_index=self.disc_batch_cnt,
                        commit=True
                    )

                critic_stepper.step()
                critic_stepper.optimizer.zero_grad()

                update_normalizers_stats(
                    gan_model.discriminator,
                    disc_real_vals=disc_real_vals,
                    disc_gen_vals=disc_gen_vals
                )

                last_disc_real_vals = disc_real_vals
                last_disc_gen_vals = disc_gen_vals

            critic_loss_total += loss.item() * len(gen_batch_x)
            gan_model.generator.requires_grad_(True)

            # =========================
            # generator training
            # =========================
            gan_model.discriminator.requires_grad_(False)

            gen_batch_x, real_batch_x, gen_batch_y, real_batch_y = get_batches(generator_batch)

            observations = (
                gan_model.discriminator(real_batch_x, real_batch_y)
                - gan_model.discriminator(gen_batch_x, gen_batch_y)
            )

            adv_gen_loss = observations.mean() # состязательный loss генератора

            fake_energy = gen_batch_x.sum(dim=tuple(range(1, gen_batch_x.ndim)))
            real_energy = real_batch_x.sum(dim=tuple(range(1, real_batch_x.ndim)))


            energy_loss = torch.mean((fake_energy - real_energy) ** 2) # суммарная энергия правдоподобная для каждого конкретного примера

            sparsity_loss = gen_batch_x.abs().mean() # sparsity = разреженность. отвечает за то, чтобы энергия была сконцентрирована в одном месте, а не размазана по всему калориметру
            # уменьшаем sparsity_loss => одинаково уменьшаем модуль каждого числа => маленькие числа превращаются в 0
            layer_frac_loss = layer_fraction_loss(real_batch_x, gen_batch_x) # распределение энергии по слоям правдоподобное для конкретного примера
            quantile_loss = quantile_energy_loss(real_batch_x, gen_batch_x) # распределение энергии = какой процент примеров из датасета имеет маленькую/среднюю/большую энергию?
            # есть настоящее распределение (20%/60%/20%) и мы его приблежаем. для этого приблежаем распределение энергии к настоящему в каждом батче

            gen_loss = ( # общий loss генератора
                adv_gen_loss
                + self.lambda_energy * energy_loss
                + self.lambda_sparsity * sparsity_loss
                + self.lambda_layer_fraction * layer_frac_loss
                + self.lambda_quantile * quantile_loss
            )
            self.gen_batch_cnt += 1

            gen_adv_loss_total += adv_gen_loss.item() * len(gen_batch_x)
            gen_energy_loss_total += energy_loss.item() * len(gen_batch_x)
            gen_sparsity_loss_total += sparsity_loss.item() * len(gen_batch_x)
            gen_layer_fraction_loss_total += layer_frac_loss.item() * len(gen_batch_x)
            gen_quantile_loss_total += quantile_loss.item() * len(gen_batch_x)

            gen_loss_total += gen_loss.item() * len(gen_batch_x)

            gen_loss.backward()

            gen_grad_norm = calc_grad_norm(gan_model.generator)
            gen_grad_norm_total += gen_grad_norm

            generator_stepper.step()
            generator_stepper.optimizer.zero_grad()

            gan_model.discriminator.requires_grad_(True)

            if logger is not None:
                logger.log_metrics(
                    data={
                        'train/generator/batch_loss': gen_loss.item(),
                        'train/generator/batch_adv_loss': adv_gen_loss.item(),
                        'train/generator/batch_energy_loss': energy_loss.item(),
                        'train/generator/batch_weighted_energy_loss': (
                            self.lambda_energy * energy_loss
                        ).item(),
                        'train/generator/batch_sparsity_loss': sparsity_loss.item(),
                        'train/generator/batch_weighted_sparsity_loss': (
                            self.lambda_sparsity * sparsity_loss
                        ).item(),
                        'train/generator/batch_layer_fraction_loss': layer_frac_loss.item(),
                        'train/generator/batch_weighted_layer_fraction_loss': (
                            self.lambda_layer_fraction * layer_frac_loss
                        ).item(),
                        'train/generator/batch_quantile_loss': quantile_loss.item(),
                        'train/generator/batch_weighted_quantile_loss': (
                            self.lambda_quantile * quantile_loss
                        ).item(),
                        'train/generator/batch_grad_norm': gen_grad_norm,
                        'train/generator/fake_energy_mean': fake_energy.mean().item(),
                        'train/generator/real_energy_mean': real_energy.mean().item(),
                    },
                    period='gen_batch',
                    period_index=self.gen_batch_cnt,
                    commit=True
                )

        if logger is not None:
            logger.log_metrics(
                data={
                    'train/critic/loss': critic_loss_total / len(train_dataset),
                    'train/critic/adv_loss': critic_adv_loss_total / len(train_dataset), # в 5 раз больше нужного
                    'train/generator/loss': gen_loss_total / len(train_dataset),
                    'train/generator/adv_loss': gen_adv_loss_total / len(train_dataset),
                    'train/generator/energy_loss': gen_energy_loss_total / len(train_dataset),
                    'train/generator/weighted_energy_loss': (
                        self.lambda_energy * gen_energy_loss_total / len(train_dataset)
                    ),
                    'train/generator/sparsity_loss': gen_sparsity_loss_total / len(train_dataset),
                    'train/generator/weighted_sparsity_loss': (
                        self.lambda_sparsity * gen_sparsity_loss_total / len(train_dataset)
                    ),
                    'train/generator/layer_fraction_loss': gen_layer_fraction_loss_total / len(train_dataset),
                    'train/generator/weighted_layer_fraction_loss': (
                        self.lambda_layer_fraction * gen_layer_fraction_loss_total / len(train_dataset)
                    ),
                    'train/generator/quantile_loss': gen_quantile_loss_total / len(train_dataset),
                    'train/generator/weighted_quantile_loss': (
                        self.lambda_quantile * gen_quantile_loss_total / len(train_dataset)
                    ),
                    'train/discriminator/grad_norm': disc_grad_norm_total / (
                        len(dataloader) * self.n_critic
                    ),
                    'train/generator/grad_norm': gen_grad_norm_total / len(dataloader),
                },
                period='epoch',
                commit=False
            )

        avg_loss_d = critic_loss_total / len(train_dataset)
        avg_loss_g = gen_loss_total / len(train_dataset)
        avg_adv_g = gen_adv_loss_total / len(train_dataset)
        avg_energy_loss = gen_energy_loss_total / len(train_dataset)
        avg_weighted_energy_loss = self.lambda_energy * avg_energy_loss
        avg_sparsity_loss = gen_sparsity_loss_total / len(train_dataset)
        avg_weighted_sparsity_loss = self.lambda_sparsity * avg_sparsity_loss
        avg_layer_fraction_loss = gen_layer_fraction_loss_total / len(train_dataset)
        avg_weighted_layer_fraction_loss = self.lambda_layer_fraction * avg_layer_fraction_loss
        avg_quantile_loss = gen_quantile_loss_total / len(train_dataset)
        avg_weighted_quantile_loss = self.lambda_quantile * avg_quantile_loss

        avg_wd = -avg_loss_d

class GanTrainer:
    def __init__(self, model_dir: ModelDir, save_checkpoint_once_in_epoch: int = 1,
                 use_saved_checkpoint: bool = True) -> None:
        self.model_dir = model_dir
        self.save_checkpoint_once_in_epoch = save_checkpoint_once_in_epoch
        self.use_saved_checkpoint = use_saved_checkpoint

    def train(self, gan_model: GAN,
              train_dataset: torch.utils.data.Dataset, val_dataset: torch.utils.data.Dataset,
              generator_stepper: Stepper, critic_stepper: Stepper,
              epoch_trainer: GanEpochTrainer,
              n_epochs: int = 100,
              metric: Optional[Metric] = None,
              metric_predicate: Optional[TrainPredicate] = None,
              logger_cm_fn: Optional[Callable[[], ContextManager[GANLogger]]] = None
              ) -> Generator[Tuple[int, GAN], None, GAN]:
        gan_model.to(get_local_device())
        inverse_to_initial_domain_fn = getattr(train_dataset, 'inverse_transform', None)
        epoch = 1

        if self.use_saved_checkpoint:
            checkpoint = self.model_dir.get_checkpoint_state()
            if checkpoint is not None:
                epoch = checkpoint['epoch']
                print(f"Checkpoint was loaded. Current epoch: {epoch}")
                gan_model.load_state_dict(checkpoint['gan'])
                generator_stepper.load_state_dict(checkpoint['generator_stepper'])
                critic_stepper.load_state_dict(checkpoint['critic_stepper'])

        if logger_cm_fn is None:
            logger_cm = contextlib.nullcontext(None)
        else:
            logger_cm = logger_cm_fn() or contextlib.nullcontext(None)

        with logger_cm as logger:
            while epoch <= n_epochs:
                if logger is not None:
                    logger.log_metrics(data={}, period='epoch', period_index=epoch, commit=False)

                epoch_trainer.train_epoch(
                    gan_model=gan_model,
                    train_dataset=train_dataset,
                    generator_stepper=generator_stepper,
                    critic_stepper=critic_stepper,
                    logger=logger
                )

                if logger is not None:
                    if metric is not None and (metric_predicate is None or metric_predicate(epoch=epoch)):
                        with torch.no_grad():
                            metrics_results = metric(
                                gan_model=gan_model,
                                train_dataset=train_dataset,
                                val_dataset=val_dataset,
                                inverse_to_initial_domain_fn=inverse_to_initial_domain_fn
                            )
                        log_metric(
                            metric,
                            results=metrics_results,
                            logger=logger,
                            period='epoch',
                            period_index=epoch
                        )

                    logger.commit(period='epoch')

                epoch += 1

                if self.save_checkpoint_once_in_epoch != 0 and epoch % self.save_checkpoint_once_in_epoch == 0:
                    checkpoint = {
                        'epoch': epoch,
                        'gan': gan_model.state_dict(),
                        'generator_stepper': generator_stepper.state_dict(),
                        'critic_stepper': critic_stepper.state_dict()
                    }
                    self.model_dir.save_checkpoint_state(checkpoint)

                yield epoch, gan_model

        return gan_model
