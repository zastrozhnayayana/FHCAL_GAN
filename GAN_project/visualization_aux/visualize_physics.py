"""
Auxiliary functions to easily visualize the results of GANs applied to the Physics task
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from visualization_aux.common import imshow


def energy_imshow(energy, ax=None, cmap='inferno', log_transform: bool = True, layer=None):
    if torch.is_tensor(energy):
        data = energy.detach().cpu().numpy()
    else:
        data = np.array(energy)
    
    # Приводим к формату (слои, y, x)
    if data.shape == (7, 5, 7):
        data = np.transpose(data, (2, 0, 1))
    elif data.shape != (7, 7, 5):
        raise ValueError(f"Ожидалась форма (7,7,5) или (7,5,7), получена {data.shape}")
    
    if log_transform:
        data = np.log1p(data)
    
    # СЛУЧАЙ 1: Показать конкретный слой
    if layer is not None:
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        im = ax.imshow(data[layer, :, :], cmap=cmap, interpolation='none')
        ax.set_title(f'Слой {layer+1}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, label='Энергия')
        return im
    
    # СЛУЧАЙ 2: Передан отдельный ax (используется в gen_several_images)
    # Показываем сумму по слоям
    if ax is not None:
        if hasattr(ax, '__len__') and not isinstance(ax, plt.Axes):
            fig, axes = plt.subplots(2, 4, figsize=(14, 7))
            vmax = np.max(data)
            
            for layer in range(7):
                ax_layer = axes[layer // 4, layer % 4]
                im = ax_layer.imshow(data[layer, :, :], cmap=cmap, interpolation='none', vmin=0, vmax=vmax)
                ax_layer.set_title(f'Слой {layer+1}')
                ax_layer.axis('off')
            
            plt.colorbar(im, ax=axes, label='Энергия', fraction=0.02, pad=0.15)
            plt.suptitle(f'log1p: {log_transform}')
            plt.tight_layout()
            plt.show()
            return fig
        else:
            data_2d = np.sum(data, axis=0)
            im = ax.imshow(data_2d, cmap=cmap, interpolation='none')
            ax.set_title('Сумма по слоям')
            ax.axis('off')
            return im
    
    # СЛУЧАЙ 3: ax=None и layer=None - показываем все слои
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    vmax = np.max(data)
    
    for layer_idx in range(7):
        ax_layer = axes[layer_idx // 4, layer_idx % 4]
        im = ax_layer.imshow(data[layer_idx, :, :], cmap=cmap, interpolation='none', vmin=0, vmax=vmax)
        ax_layer.set_title(f'Слой {layer_idx+1}')
        ax_layer.axis('off')
    
    plt.colorbar(im, ax=axes, label='Энергия', fraction=0.02, pad=0.15)
    plt.suptitle(f'log1p: {log_transform}')
    plt.tight_layout()
    plt.show()
    return fig
    # if torch.is_tensor(energy):
    #     data = energy.detach().cpu().numpy()
    # else:
    #     data = np.array(energy)
    
    # if data.shape == (7, 9, 10):
    #     data = np.transpose(data, (2, 0, 1))
    
    # if log_transform:
    #     data = np.log1p(data)
    
    # if layer is not None:
    #     if ax is None:
    #         fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    #     im = ax.imshow(data[layer, :, :], cmap=cmap, interpolation='none')
    #     ax.set_title(f'Слой {layer+1}')
    #     ax.axis('off')
    #     plt.colorbar(im, ax=ax, label='Энергия')
    #     return im
    
    # if ax is None:
    #     fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    # else:
    #     axes = ax
    
    # vmax = np.max(data)
    # for layer in range(10):
    #     ax_layer = axes[layer // 5, layer % 5]
    #     im = ax_layer.imshow(data[layer, :, :], cmap=cmap, interpolation='none', vmin=0, vmax=vmax)
    #     ax_layer.set_title(f'Слой {layer+1}')
    #     ax_layer.axis('off')
    
    # plt.colorbar(im, ax=axes, label='Энергия', fraction=0.02, pad=0.15)
    # plt.suptitle(f'log1p: {log_transform}')
    # plt.tight_layout()
    # plt.show()


def add_noise(arr):
    noise_coefs = 1 + np.random.normal(0, 0.1, size=arr.shape)
    return arr * noise_coefs


def get_test_data(global_config):
    """
    Примеры данных для визуалиации (отбирались вручную)
    """
    data_train = np.load(os.path.join(global_config.paths.data_dir_path,
                                      'fhcal_data3.npz'))
    samples_indices = [0, 2, 4, 10, 20]

    pure_points = torch.Tensor(
        data_train['ParticlePoint'][samples_indices, :2]
    )
    pure_momentums = torch.Tensor(
        data_train['ParticleMomentum'][samples_indices]
    )

    noised_points = torch.Tensor(add_noise(
        data_train['ParticlePoint'][samples_indices, :2]
    ))
    noised_momentums = torch.Tensor(add_noise(
        data_train['ParticleMomentum'][samples_indices]
    ))

    energy = data_train['EnergyDeposit'][samples_indices]
    points = torch.vstack([pure_points, noised_points])
    momentums = torch.vstack([pure_momentums, noised_momentums])

    return energy, points, momentums


    

   
    
def plot_fhcal_simple(energy_data, event_idx=None):
    
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    
    # Конвертируем тензор в numpy
    if isinstance(energy_data, torch.Tensor):
        data = energy_data.detach().cpu().numpy()
    else:
        data = np.array(energy_data)
    
    # Приводим к единому формату (y, x, слои)
    if data.shape == (7, 7, 5):
        data = np.transpose(data, (1, 2, 0))
    elif data.shape == (7, 5, 7):
        pass
    else:
        raise ValueError(f"Ожидалась форма (10,7,9) или (7,9,10), получена {data.shape}")
    
 
    data = np.array(data)
    

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    vmax = np.max(data)
    
    for layer in range(7):
        ax = axes[layer // 4, layer % 4]
        layer_data = data[:, :, layer]
        
        im = ax.imshow(layer_data, cmap='inferno', interpolation='none', vmin=0, vmax=vmax)
        ax.set_title(f'Слой {layer+1}')
        ax.axis('off')
    
    # Colorbar
    plt.colorbar(im, ax=axes, label='Энергия', fraction=0.02, pad=0.15)
    
    total = np.sum(data)
    title = f'Событие {event_idx}, энергия: {total:.2f}' if event_idx is not None else f'Энергия: {total:.2f}'
    plt.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    # plt.show()
    # return fig
  