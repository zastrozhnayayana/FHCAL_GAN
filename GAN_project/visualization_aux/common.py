import numpy as np
import matplotlib.pyplot as plt
import torch

from pipeline.device import get_local_device
from pipeline.gan import GAN


def imshow(img, ax=None, cmap='viridis', affine: bool = True):
    npimg = img.detach().numpy()
    if affine:  # если изначально к изображениям применялось это преобразование
        npimg = npimg / 2 + 0.5  # обратное афинное преобразование
    if ax is None:
        if npimg.shape[0] == 1:  # 1 channel
            plt.imshow(np.squeeze(npimg), cmap=cmap)
        else:  # 3 channels
            plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap=cmap)
        plt.show()
    else:
        if npimg.shape[0] == 1:  # 1 channel
            ax.imshow(np.squeeze(npimg), cmap=cmap)
        else:  # 3 channels
            ax.imshow(np.transpose(npimg, (1, 2, 0)), cmap=cmap)


# def gen_several_images(gan_model: GAN, n: int = 5, y=None, figsize=(13, 13), imshow_fn=imshow):
#     """
#     Выводит n изображений, сгенерированных gan_model в строке
#     """
#     fig, axs = plt.subplots(nrows=1, ncols=n, figsize=figsize)
#     gan_model.to(get_local_device())
#     with torch.no_grad():
#         noise_batch = gan_model.gen_noise(n).to(get_local_device())
#         gen_batch = gan_model.generator(noise_batch, y)

#     if n == 1:
#         axs = [axs]
#     for i, (tensor, ax) in enumerate(zip(gen_batch, axs)):
#         imshow_fn(tensor.cpu(), ax=ax)
#         if y is not None and isinstance(y, torch.Tensor):
#             ax.set_xlabel(y[i].item())

#     plt.show()
def gen_several_images(gan_model: GAN, n: int = 5, y=None, figsize=(13, 13), imshow_fn=imshow):
    """
    Выводит n изображений, сгенерированных gan_model в строке
    """
    fig, axs = plt.subplots(nrows=1, ncols=n, figsize=figsize)
    gan_model.to(get_local_device())
    with torch.no_grad():
        noise_batch = gan_model.gen_noise(n).to(get_local_device())
        
        # Если y - кортеж (point, momentum), передаем как есть
        if isinstance(y, tuple):
            gen_batch = gan_model.generator(noise_batch, y)
        else:
            gen_batch = gan_model.generator(noise_batch, y)

    if n == 1:
        axs = [axs]
    
    for i, (tensor, ax) in enumerate(zip(gen_batch, axs)):
        # Для 3D данных суммируем по слоям
        if tensor.dim() == 3:  # (10, 7, 9)
            tensor_2d = tensor.sum(dim=0)
        elif tensor.dim() == 4:  # (batch, 10, 7, 9)
            tensor_2d = tensor.sum(dim=1)
        else:
            tensor_2d = tensor
            
        imshow_fn(tensor_2d.cpu(), ax=ax)
        
        if y is not None and not isinstance(y, tuple):
            ax.set_xlabel(y[i].item() if hasattr(y[i], 'item') else str(y[i]))

    plt.show()


def visualize_last_event(gan, y_point, y_momentum, epoch=None, figsize=(18, 8)):
    """
    Визуализация одного сгенерированного события (все 10 слоев)
    """
    import matplotlib.pyplot as plt
    import torch
    from pipeline.device import get_local_device

    gan.to(get_local_device())

    # Запоминаем, был ли GAN в режиме train
    was_training = gan.training

    # Для BatchNorm при batch_size=1 нужен eval mode
    gan.eval()

    try:
        with torch.no_grad():
            noise = gan.gen_noise(1).to(get_local_device())
            fake_data = gan.generator(noise, (y_point, y_momentum))  # (1, 10, 7, 9)

        event_data = fake_data[0].cpu().numpy()  # (10, 7, 9)

        fig, axes = plt.subplots(2, 5, figsize=figsize)
        vmax = event_data.max()

        for layer in range(7):
            ax = axes[layer // 5, layer % 5]
            layer_data = event_data[layer, :, :]

            im = ax.imshow(
                layer_data,
                cmap='inferno',
                interpolation='none',
                vmin=0,
                vmax=vmax
            )
            ax.set_title(f'Слой {layer+1}')
            ax.axis('off')

        plt.colorbar(im, ax=axes, label='Энергия', fraction=0.02, pad=0.15)

        total_energy = event_data.sum()
        if epoch is not None:
            plt.suptitle(f'Эпоха {epoch}, энергия: {total_energy:.2f}', fontsize=14)
        else:
            plt.suptitle(f'Энергия: {total_energy:.2f}', fontsize=14)

        plt.tight_layout()
        plt.show()

        return fig

    finally:
        # Возвращаем GAN в train mode, если он был train до визуализации
        if was_training:
            gan.train()