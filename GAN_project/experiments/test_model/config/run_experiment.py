"""
This file is expecting the 'pipeline' package name to be defined
This file completely defines the experiment to run
"""
from typing import Tuple, Generator, Optional, Dict, List

import numpy as np
import torch
import torch.utils.data

from pipeline import data
from pipeline import logger
from pipeline.discriminators import SimplePhysicsDiscriminator, CaloganPhysicsDiscriminator
from pipeline.evaluation import evaluate_model
from pipeline.experiment_setup import experiments_storage, global_config, init_logger
from pipeline.gan import GAN
from pipeline.generators import SimplePhysicsGenerator, CaloganPhysicsGenerator
from pipeline.metrics import *
from pipeline.config import load_global_config
from pipeline.custom_metrics import *
from pipeline.normalization import apply_normalization, SpectralNormalizer, WeakSpectralNormalizer,\
                          MultiplyOutputNormalizer, ABCASNormalizer
from pipeline.predicates import TrainPredicate, IgnoreFirstNEpochsPredicate, EachNthEpochPredicate
from pipeline.regularizer import *
from pipeline.results_storage import ResultsStorage
from pipeline.storage import ExperimentsStorage
from pipeline.train import Stepper, WganEpochTrainer, GanTrainer
from pipeline.comet_logger import CometCM


def form_metric() -> Metric:
    return MetricsSequence(
        # BetaMetric(),
        # DiscriminatorParameterMetric('r'),
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
    # return None


def form_result_metrics() -> Metric:
    return MetricsSequence(
                DataStatistics(
                    KLDivergence(LongitudualClusterAsymmetryMetric()),
                    KLDivergence(TransverseClusterAsymmetryMetric()),
                    KLDivergence(ClusterLongitudualWidthMetric()),
                    KLDivergence(ClusterTransverseWidthMetric()),
                ),
            )


def form_dataset(train: bool = False) -> torch.utils.data.Dataset:
    data_filepath = global_config.paths.data_dir_path + '/caloGAN_case11_5D_120K.npz'
    return data.UnifiedDatasetWrapper(data.get_physics_dataset(data_filepath, train=train))


def form_gan_trainer(model_name: str, gan_model: Optional[GAN] = None, n_epochs: int = 100) -> Generator[Tuple[int, GAN], None, GAN]:
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
    # discriminator = apply_normalization(discriminator, MultiplyOutputNormalizer, coef=2., is_trainable_coef=False)
    # discriminator = apply_normalization(discriminator, WeakSpectralNormalizer, beta=2., is_trainable_beta=False)
    # discriminator = apply_normalization(discriminator, ABCASNormalizer)

    regularizer = BasicRegularizer(lambda: sum(p.norm(2) for p in discriminator.parameters()))
    regularizer = PowRegularizer(regularizer, 2)
    regularizer = MultiplierRegularizer(regularizer, start_value=1.)

    # lambd = 1.
    # normalization_loss = lambda: lambd*discriminator.coef**2

    if gan_model is None:
        gan_model = GAN(generator, discriminator, uniform_noise_generator)

    generator_stepper = Stepper(
        optimizer=torch.optim.RMSprop(generator.parameters(), lr=1e-3)
    )

    discriminator_stepper = Stepper(
        optimizer=torch.optim.RMSprop(discriminator.parameters(), lr=1e-5)
    )

    epoch_trainer = WganEpochTrainer(n_critic=5, batch_size=100)

    model_dir = experiments_storage.get_model_dir(model_name)
    trainer = GanTrainer(model_dir=model_dir, use_saved_checkpoint=True, save_checkpoint_once_in_epoch=5)
    train_gan_generator = trainer.train(gan_model=gan_model,
                                        train_dataset=train_dataset, val_dataset=val_dataset,
                                        generator_stepper=generator_stepper,
                                        critic_stepper=discriminator_stepper,
                                        epoch_trainer=epoch_trainer,
                                        n_epochs=n_epochs,
                                        metric=metric, metric_predicate=metric_predicate,
                                        logger_cm_fn=logger_cm_fn,
                                        regularizer=regularizer)
    return train_gan_generator, epoch_trainer


def run() -> GAN:
    model_name = 'physics_test'
    gan_trainer, epoch_trainer = form_gan_trainer(model_name=model_name, n_epochs=60)
    gan = None
    for epoch, gan in gan_trainer:
        pass
    loss_arr = epoch_trainer.get_loss_arr()
    print(loss_arr)
    # evaluate model somehow ...
    return gan


if __name__ == '__main__':
    run()
