import torch.utils.data

from pipeline.gan import GAN
from pipeline.logger import GANLogger
from pipeline.metrics import Metric, unravel_metric_results


def evaluate_model(gan_model: GAN, val_dataset: torch.utils.data.Dataset,
                   metric: Metric, logger: GANLogger) -> None:
    metric_results = metric.evaluate(gan_model=gan_model, val_dataset=val_dataset)

    new_data = unravel_metric_results(metric, metric_results)
    logger.log_summary_metrics(new_data)
