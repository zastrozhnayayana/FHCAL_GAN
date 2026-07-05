# Taken from: https://github.com/SchattenGenie/mlhep2019_2_phase/blob/master/analysis
import pathlib
from typing import Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import auc
from torch import nn
from tqdm import tqdm

from .prd_score import compute_prd_from_embedding


class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()

        self.batchnorm0 = nn.BatchNorm2d(7)

        self.conv1 = nn.Conv2d(7, 16, 3, stride=2, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(64)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.1)

        self.fc1 = nn.Linear(64, 256)
        self.batchnorm4 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)

        self.fc4 = nn.Linear(64, 5)  # x, y, px, py, pz
        self.fc5 = nn.Linear(64, 1)  # total energy
        self.fc6 = nn.Linear(64, 7)  # layer fractions

    def _encode_base(self, x):
        x = self.batchnorm0(x)

        x = self.batchnorm1(F.relu(self.conv1(x)))
        x = self.batchnorm2(F.relu(self.conv2(x)))
        x = self.batchnorm3(F.relu(self.conv3(x)))

        x = self.adaptive_pool(x)
        x = x.view(len(x), -1)

        x = self.dropout(x)
        x = self.batchnorm4(F.relu(self.fc1(x)))
        x = F.leaky_relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))

        return x

    def forward(self, x):
        encoding = self._encode_base(x)

        pred_condition = self.fc4(encoding)
        pred_energy = self.fc5(encoding)
        pred_layer_frac = F.softmax(self.fc6(encoding), dim=1)

        return pred_condition, pred_energy, pred_layer_frac

    def get_encoding(self, x):
        return self._encode_base(x)


def load_embedder(state_path: str):
    embedder = Regressor()
    try:
        checkpoint = torch.load(state_path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            embedder.load_state_dict(checkpoint['model_state'])
        else:
            embedder.load_state_dict(checkpoint)
        print("✅ Загружен обученный эмбеддер")
    except FileNotFoundError:
        print("⚠️ Файл весов не найден, используется необученный эмбеддер")

    embedder.eval()
    return embedder


embedder_state_path = pathlib.Path(__file__).parent / pathlib.Path(
    './embedder_state_7x5_xy_energy_frac.pt'
)
embedder = load_embedder(str(embedder_state_path))


def get_energy_embedding(data):
    x = data[0].view(-1, 7, 7, 5)
    return embedder.get_encoding(x).detach().numpy(), data[1]


def check_tensor_is_finite(t: np.ndarray) -> torch.Tensor:
    x = t.astype(np.float64)

    if np.isnan(x).sum() > 0:
        print('tensor contains NaN')
    if np.isinf(x).sum() > 0:
        print('tensor contains inf')
    if np.isneginf(x).sum() > 0:
        print('tensor contains -inf')

    mask = np.isnan(x) | np.isinf(x) | np.isneginf(x)

    bad_rows = mask.sum(axis=1) != 0
    bad_rows_cnt = bad_rows.sum()

    if bad_rows_cnt != 0:
        print(f'{bad_rows_cnt} bad rows in tensor')

    return x[~bad_rows]


def calc_pr_rec_from_embeds(
    data_real_embeds: np.ndarray,
    data_fake_embeds: np.ndarray,
    num_clusters=20,
    num_runs=10,
    NUM_RUNS=10,
    show_progress_bar: bool = False,
    enforce_balance: bool = True
) -> Tuple[List[np.ndarray], List[np.ndarray]]:

    min_len = min(len(data_real_embeds), len(data_fake_embeds))
    data_real_embeds = data_real_embeds[:min_len]
    data_fake_embeds = data_fake_embeds[:min_len]

    data_real_embeds = check_tensor_is_finite(data_real_embeds)
    data_fake_embeds = check_tensor_is_finite(data_fake_embeds)

    precisions = []
    recalls = []

    iterator = tqdm(range(NUM_RUNS)) if show_progress_bar else range(NUM_RUNS)

    for _ in iterator:
        precision, recall = compute_prd_from_embedding(
            data_real_embeds,
            data_fake_embeds,
            num_clusters=num_clusters,
            num_runs=num_runs,
            enforce_balance=enforce_balance
        )
        precisions.append(precision)
        recalls.append(recall)

    return precisions, recalls


def plot_pr_aucs(precisions: List[np.ndarray], recalls: List[np.ndarray]):
    plt.figure(figsize=(12, 12))
    pr_aucs = []

    for p, r in zip(precisions, recalls):
        valid = np.isfinite(p) & np.isfinite(r)
        p = p[valid]
        r = r[valid]

        if len(r) < 2:
            pr_aucs.append(0.0)
            continue

        idx = np.argsort(r)
        r_sorted = r[idx]
        p_sorted = p[idx]

        unique_recalls = []
        unique_precisions = []

        i = 0
        while i < len(r_sorted):
            j = i
            while j < len(r_sorted) and r_sorted[j] == r_sorted[i]:
                j += 1

            unique_recalls.append(r_sorted[i])
            unique_precisions.append(np.max(p_sorted[i:j]))
            i = j

        if len(unique_recalls) < 2:
            pr_aucs.append(0.0)
            continue

        try:
            auc_val = auc(unique_recalls, unique_precisions)
        except ValueError:
            auc_val = np.trapz(unique_precisions, unique_recalls)

        pr_aucs.append(auc_val)
        plt.step(r, p, color='b', alpha=0.2)

    if len(recalls) > 0:
        mean_recall = np.mean(recalls, axis=0)
        mean_precision = np.mean(precisions, axis=0)

        sort_idx = np.argsort(mean_recall)
        mean_recall_sorted = mean_recall[sort_idx]
        mean_precision_sorted = mean_precision[sort_idx]

        plt.step(
            mean_recall_sorted,
            mean_precision_sorted,
            color='r',
            alpha=1,
            label=f'average, PR-AUC={np.mean(pr_aucs):.4f}'
        )

        std_precision = np.std(precisions, axis=0)

        plt.fill_between(
            mean_recall_sorted,
            mean_precision_sorted - std_precision[sort_idx] * 3,
            mean_precision_sorted + std_precision[sort_idx] * 3,
            color='g',
            alpha=0.2,
            label='std'
        )

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend()
    plt.title('PRD')

    return pr_aucs