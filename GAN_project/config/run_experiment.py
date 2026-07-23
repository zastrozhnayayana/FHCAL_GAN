"""
This file is expecting the 'pipeline' package name to be defined
This file completely defines the experiment to run
"""
from typing import Tuple, Generator, Optional

import numpy as np
import torch
import torch.utils.data

from pipeline import data
from pipeline.discriminators import CaloganPhysicsDiscriminator
from pipeline.evaluation import evaluate_model # ПОКА НИКАК НЕ ОЦЕНИВАЕМ МОДЕЛЬ
from pipeline.experiment_setup import experiments_storage, global_config, init_logger
from pipeline.gan import GAN
from pipeline.generators import CaloganPhysicsGenerator
from pipeline.metrics import *
from pipeline.custom_metrics import *
from pipeline.normalization import apply_normalization, SpectralNormalizer
from pipeline.predicates import TrainPredicate, IgnoreFirstNEpochsPredicate, EachNthEpochPredicate
from pipeline.train import Stepper, WganEpochTrainer, GanTrainer


def form_metric() -> Metric:
    return MetricsSequence(
        CriticValuesDistributionMetric(values_cnt=1000),
        PhysicsDataStatistics(
            *[statistic_cls() for statistic_cls in PHYS_STATISTICS],
            create_prd_energy_embed(),
            create_conditional_prd_energy_embed(),
            create_prd_physics_statistics(),
            create_conditional_prd_physics_statistics(),
        ),
    )


def form_metric_predicate() -> Optional[TrainPredicate]:
    return IgnoreFirstNEpochsPredicate(20) & EachNthEpochPredicate(5)


def form_dataset(train: bool = False) -> torch.utils.data.Dataset:
    data_filepath = global_config.paths.data_dir_path + '/fhcal_data3.npz'
    return data.UnifiedDatasetWrapper(data.get_physics_dataset(data_filepath, train=train))


def form_gan_trainer(model_name: str, n_epochs: int = 100) -> Generator[Tuple[int, GAN], None, GAN]:
    """
    :return: a generator that yields (epoch number, gan_model after this epoch)
    """
    logger_cm_fn = init_logger(model_name)
    metric = form_metric()
    metric_predicate = form_metric_predicate()

    train_dataset = form_dataset(train=True)
    val_dataset = form_dataset(train=False)

    # for local testing
    val_size = int(0.1 * len(val_dataset))
    val_dataset = torch.utils.data.Subset(val_dataset, np.arange(val_size))
    # -------
    noise_dimension = 50

    def uniform_noise_generator(n: int) -> torch.Tensor:
        return 2*torch.rand(size=(n, noise_dimension)) - 1  # [-1, 1]

    generator = CaloganPhysicsGenerator(noise_dim=noise_dimension)
    discriminator = CaloganPhysicsDiscriminator()
    discriminator = apply_normalization(discriminator, SpectralNormalizer)

    gan_model = GAN(generator, discriminator, uniform_noise_generator)

    generator_stepper = Stepper(
        optimizer=torch.optim.RMSprop(generator.parameters(), lr=1e-4)
    )

    discriminator_stepper = Stepper(
        optimizer=torch.optim.RMSprop(discriminator.parameters(), lr=1e-4)
    )

    # пять обновлений критика на одно обновление генератора
    epoch_trainer = WganEpochTrainer(n_critic=5, batch_size=100) # воспроизводит одну эпоху обучения

    model_dir = experiments_storage.get_model_dir(model_name)
    trainer = GanTrainer(model_dir=model_dir, use_saved_checkpoint=True, save_checkpoint_once_in_epoch=5) # воспроизводит полный цикл обучения
    train_gan_generator = trainer.train(gan_model=gan_model,
                                        train_dataset=train_dataset, val_dataset=val_dataset,
                                        generator_stepper=generator_stepper,
                                        critic_stepper=discriminator_stepper,
                                        epoch_trainer=epoch_trainer,
                                        n_epochs=n_epochs,
                                        metric=metric, metric_predicate=metric_predicate,
                                        logger_cm_fn=logger_cm_fn)
    return train_gan_generator, epoch_trainer
    # train_gan_generator — объект, который запускает обучение по эпохам.
    # epoch_trainer — объект, который непосредственно знает, как обучать WGAN в течение одной эпохи и хранит историю loss.

def run() -> GAN:
    model_name = 'physics_test'
    gan_trainer, epoch_trainer = form_gan_trainer(model_name=model_name, n_epochs=30)
    gan = None
    for epoch, gan in gan_trainer:
        pass
    loss_arr = epoch_trainer.get_loss_arr()
    print(loss_arr)
    # evaluate model somehow ...
    return gan


if __name__ == '__main__':
    run()
