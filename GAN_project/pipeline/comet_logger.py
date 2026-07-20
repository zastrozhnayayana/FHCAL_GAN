import tempfile
from copy import copy
from typing import Any, Dict, Optional, Sequence

import numpy as np
from comet_ml import Experiment
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde

from pipeline.logger import GANLogger, LoggerConfig
from pipeline.run_env import get_comet_token


class _CometLogger(GANLogger):
    """Comet ML implementation of :class:`GANLogger`.

    This class is intended to be created by ``CometCM`` rather than directly.
    Its public logging behaviour mirrors ``_WandbLogger``.
    """

    COMET_POINTS_LIMIT = 10_000 # Максимальное количество точек, которое можно залогировать в comet за один раз
    PYPLOT_FORMAT = 'png'
    SUMMARY_METRICS_TABLE_NAME = 'summary_metrics'

    def __init__(self, config: Optional[LoggerConfig] = None,
                 experiment: Optional[Experiment] = None) -> None:
        super().__init__(config=config)
        self.experiment = experiment
        self.summary_metrics: Dict[str, Any] = {}

    # Выгружаем данные в comet
    def _log_metrics(self, data: Dict[str, Any], period: str,
                     period_index: int) -> None:
        logged_data = copy(data)
        logged_data[period] = period_index

        kwargs = {'step': period_index}
        if period == 'epoch':
            kwargs['epoch'] = period_index
        self.experiment.log_metrics(logged_data, **kwargs) # Выгружаем данные в comet

    @staticmethod
    # Если превышено количество точек, то случайным образом выбираем limit точек из values
    def _sample(values: Sequence[float], limit: int) -> np.ndarray:
        values = np.asarray(values).reshape(-1)
        if len(values) > limit:
            indices = np.random.choice(len(values), limit, replace=False)
            values = values[indices]
        return values

    # Логирует распределения разных значений на одном графике
    def log_distribution(self, values: Dict[str, Sequence[float]], name: str,
                         period: str, period_index: int,
                         log_histogram: bool = True) -> None:
        if not values:
            return

        values_per_key = max(1, self.COMET_POINTS_LIMIT // len(values))
        # Не слишком много точек хотим логировать
        sampled_values = {
            key: self._sample(vals, values_per_key)
            for key, vals in values.items()
        }

        if log_histogram:
            for key, vals in sampled_values.items():
                self.experiment.log_histogram_3d(
                    vals,
                    name=f'{name}/{key}',
                    step=period_index,
                    epoch=period_index if period == 'epoch' else None,
                )
            return

        # Если хотим построить KDE кривые
        non_empty_values = [vals for vals in sampled_values.values() if len(vals)]
        if not non_empty_values:
            return

        min_x = min(float(np.min(vals)) for vals in non_empty_values)
        max_x = max(float(np.max(vals)) for vals in non_empty_values)
        xs = np.linspace(min_x - 0.1, max_x + 0.1, num=100)

        for key, vals in sampled_values.items():
            if len(vals) < 2 or np.all(vals == vals[0]):
                continue
            ys = gaussian_kde(vals)(xs)
            self.experiment.log_curve(
                name=f'{name}/{key}',
                x=xs,
                y=ys,
                step=period_index,
            )

    # Логирует распределение значений критика для настоящих и сгенерированных данных на одном графике    
    def log_critic_values_distribution(self,
                                       critic_values_true: Sequence[float],
                                       critic_values_gen: Sequence[float],
                                       period: str,
                                       period_index: int) -> None:
        self.log_distribution(
            values={
                'gen': critic_values_gen,
                'true': critic_values_true,
            },
            name='Critic values',
            period=period,
            period_index=period_index,
        )

    # Сохраняет картинку в comet
    def log_pyplot(self, name: str, period: str, period_index: int) -> None:
        # Keep the temporary-file workflow used by the W&B logger so callers do
        # not need to know which backend is active.
        with tempfile.NamedTemporaryFile(suffix=f'.{self.PYPLOT_FORMAT}') as file:
            plt.savefig(file, format=self.PYPLOT_FORMAT)
            file.seek(0)
            self.experiment.log_image(
                file,
                name=name,
                step=period_index,
            )
        plt.close()


    def log_summary_metrics(self, data: Dict[str, Any]) -> None:
        """Update the single-row min/max/last summary used by ``GANLogger``."""
        self.summary_metrics.update(data)

        # Log individual values as experiment metadata for convenient lookup.
        for metric_name, value in data.items():
            self.experiment.log_other(metric_name, value)

        # Also retain the W&B-like one-row summary table representation.
        # Выгружаем таблицу {название статистики (min/max/last) + название метрики (всякие лоссы): значение статистики (номер эпохи)} в comet
        self.experiment.log_table(
            filename=f'{self.SUMMARY_METRICS_TABLE_NAME}.csv',
            tabular_data=[list(self.summary_metrics.values())],
            headers=list(self.summary_metrics.keys()),
        )


# Класс, который управляет жизненным циклом эксперимента Comet ML.
class CometCM:
    """Context manager that owns a Comet ML experiment lifecycle.

    If ``token`` is omitted, Comet ML uses its normal configuration lookup,
    including the ``COMET_API_KEY`` environment variable and ``.comet.config``.
    """

    def __init__(self, project_name: str, experiment_id: str,
                 token: Optional[str] = None,
                 config: Optional[LoggerConfig] = None,
                 workspace: Optional[str] = None) -> None:
        self.project_name = project_name
        self.experiment_id = experiment_id
        if token is None:
            token = get_comet_token()
        self.token = token
        self.config = config
        self.workspace = workspace
        self.experiment: Optional[Experiment] = None

    def __enter__(self) -> _CometLogger:
        experiment_kwargs = {
            'project_name': self.project_name,
        }
        if self.token is not None:
            experiment_kwargs['api_key'] = self.token
        if self.workspace is not None:
            experiment_kwargs['workspace'] = self.workspace

        self.experiment = Experiment(**experiment_kwargs)
        self.experiment.set_name(self.experiment_id)
        return _CometLogger(config=self.config, experiment=self.experiment)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.experiment is not None:
            self.experiment.end()
