from itertools import cycle
from typing import Tuple, Union, Any, Sequence

import numpy as np
import torch
import torch.utils.data

# ВАЖНО
"""
Датасеты могут быть двух типов:
1. Элемент - число или тензор. В этом случае элемент рассматривается как x в GAN
2. Элемент - tuple длины 2. В этом случае 1-ый элемент tuple - x, 2-й - y (условие)
y - либо число, либо тензор, либо tuple с числами/тензорами

Обёртка в виде UnifiedDatasetWrapper приводит оба типа датасетов ко 2-му (для датасета 1-го типа y = None).
Обёртку следует использовать пользователю.
"""


class RandomDataloader(torch.utils.data.DataLoader):
    def __init__(self, dataset: torch.utils.data.Dataset, batch_size: int, *args, **kwargs):
        sampler = torch.utils.data.sampler.RandomSampler(dataset, replacement=True)
        random_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size=batch_size,
                                                               drop_last=False)

        super().__init__(dataset, batch_sampler=random_sampler, *args, **kwargs)


def get_random_infinite_dataloader(dataset: torch.utils.data.Dataset, batch_size: int, *args, **kwargs):
    return cycle(RandomDataloader(dataset, batch_size=batch_size, *args, **kwargs))


def collate_fn(els_list: Sequence[Union[Tuple, torch.Tensor]]):
    if isinstance(els_list[0], tuple):
        return tuple(collate_fn(list(a)) for a in zip(*els_list))
    elif isinstance(els_list[0], torch.Tensor):
        return torch.stack(tuple(els_list))
    else:
        raise RuntimeError


def stack_batches(batches_list):
    if isinstance(batches_list[0], tuple):
        return tuple(stack_batches(list(a)) for a in zip(*batches_list))
    elif isinstance(batches_list[0], torch.Tensor):
        return torch.concat(batches_list, dim=0)


def move_batch_to(batch, device):
    if isinstance(batch, tuple):
        return tuple(move_batch_to(subbatch, device) for subbatch in batch)
    else:
        return batch.to(device)


# USED
class UnifiedDatasetWrapper(torch.utils.data.Dataset):
    """
    Обёртка для поддержки датасетов обоих типов
    """
    def __init__(self, dataset: torch.utils.data.Dataset):
        self.dataset = dataset
        self.inverse_transform = getattr(dataset, 'inverse_transform', None)
        # У нас всегда есть inverse_transform, т.к. мы используем PhysicsDataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, n: int) -> Tuple[Any, Any]:
        return self.dataset[n]

# USED
class PhysicsDataset(torch.utils.data.Dataset):
    """
    one element: (energy deposit, (point, momentum))
    """
    def __init__(self, energy: torch.Tensor, point: torch.Tensor, momentum: torch.Tensor,
                 transform, inverse_transform) -> None:
        self.transform = transform
        self.inverse_transform = inverse_transform  # чтобы можно было потом восстановить исходные значения энергии

        energy = self.transform(energy)

        self.energy = energy
        self.point = point
        self.momentum = momentum

    def __getitem__(self, idx: int) -> tuple:
        # элемент датасета - кортеж из 2-х элементов: energy и tuple из point и momentum
        return self.energy[idx], (self.point[idx], self.momentum[idx])

    def __len__(self) -> int:
        return self.energy.shape[0]


# ln(1 + x)
def log1p_transform(x: torch.Tensor):
    return torch.log1p(x)

# exp(x) - 1 (обратная функция к log1p_transform)
def log1p_inverse_transform(x: torch.Tensor):
    return torch.expm1(x)


# USED
# Берёт путь к файлу с данными и возвращеает train/val датасеты
def get_physics_dataset(path: str, train: bool = True,
                        val_ratio: float = 0.5) -> torch.utils.data.Dataset:
    TRAIN_VAL_SPLIT_SEED = 0x3df3fa

    data_train = np.load(path)
    
    # рандомно выбираем индексы для валидации и обучения
    np.random.seed(TRAIN_VAL_SPLIT_SEED)
    dataset_size = len(data_train['EnergyDeposit'])
    val_size = int(dataset_size * val_ratio)

    all_indices = np.arange(dataset_size)
    val_indices = np.random.choice(all_indices, size=val_size, replace=False)
    val_mask = np.zeros(dataset_size, dtype=bool)
    val_mask[val_indices] = True
    train_indices = all_indices[~val_mask]
    indices = train_indices if train else val_indices

    # переставляем оси тензора energy и берём только x, y координаты точки частицы
    energy = torch.tensor(data_train['EnergyDeposit'][indices]).float()
    # размерность energy: (N, 7, 5, 7) - (примеры, y, x, слои)
    energy = torch.permute(energy, dims=(0, 3, 1, 2))
    # размерность energy: (N, 7, 7, 5) - (примеры, слои, y, x)
    point = torch.tensor(data_train['ParticlePoint'][:, :2][indices]).float() # z 
    momentum = torch.tensor(data_train['ParticleMomentum'][indices]).float()

    # логарифмируем энергию, чтобы распределение было более равномерным, и модель училась стабильнее
    return PhysicsDataset(energy, point, momentum,
                          transform=log1p_transform,
                          inverse_transform=log1p_inverse_transform)
