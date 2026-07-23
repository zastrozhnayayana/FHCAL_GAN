import torch


# for physics data
def add_angle_and_norm(points: torch.Tensor) -> torch.Tensor:
    angles = torch.atan2(points[:, 1], points[:, 0])[:, None] # добавляем угол в радианах (в диапазоне [-pi, pi])
    norms = torch.linalg.norm(points, dim=1)[:, None] # добавляем норму вектора
    return torch.concat([points, angles, norms], dim=1)

# L2-норма
def calc_grad_norm(model) -> float:
    total_norm = 0.
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm
