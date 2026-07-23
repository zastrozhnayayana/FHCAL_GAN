from abc import abstractmethod
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from pipeline import _aux


class Discriminator(nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor, y: Any = None) -> torch.Tensor:
        """
        :param x: object from the considered space
        :param y: condition
        None means no condition.
        A discriminator knows the exact type of condition and how to use it.
        If discriminator does not support conditions, it is expected to raise an exception.
        """
        pass

class CaloganPhysicsDiscriminator(Discriminator):
    def __init__(self, act_func=F.leaky_relu):
        super().__init__()
        self.activation = act_func

        # Свертки с stride=2 для уменьшения размера
        self.conv1 = nn.Conv2d(7, 32, 3, stride=2, padding=1)  # 7x5 -> 4x3
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)  # 4x3 -> 2x2 (ОБРЕЗАЛИ!!!)
        
        # Дополнительные свертки без уменьшения размера
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)  # 2x2 -> 2x2
        self.conv4 = nn.Conv2d(128, 256, 3, stride=1, padding=1)  # 2x2 -> 2x2
        
        # Adaptive pooling для получения 1x1
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        condition_dim = 7
        self.fc1 = nn.Linear(256 + condition_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, EnergyDeposit, y):
        point, momentum = y
        point = _aux.add_angle_and_norm(point)
        
        X = self.activation(self.conv1(EnergyDeposit))
       
        
        X = self.activation(self.conv2(X))
       
        
        X = self.activation(self.conv3(X))
       
        
        X = self.activation(self.conv4(X))
        
        
        X = self.adaptive_pool(X) # (B, 256, 1, 1)
     
        
        X = X.reshape(-1, 256)
        X = torch.cat([X, momentum, point], dim=1)
        
        X = F.leaky_relu(self.fc1(X))
        X = F.leaky_relu(self.fc2(X))
        
        return self.fc3(X)

