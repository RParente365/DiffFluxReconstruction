import torch
from typing import Optional

'''
LogStandardScaler is used to LogTransform the features and targets,
then standardize them (mean = 0, std = 1)
'''

class LogStandardScaler:
    def __init__(self, eps = 1e-8) -> None:
        self.mean: torch.Tensor = torch.empty(0)
        self.std: torch.Tensor = torch.empty(0)
        self.eps: float = eps
        self.isFitted = False

    def LogTransform(self, data: torch.Tensor) -> torch.Tensor:
        return torch.log(data + self.eps)

    def InverseLogTransform(self, data: torch.Tensor) -> torch.Tensor:
        return torch.exp(data) - self.eps

    def fit(self, data: torch.Tensor) -> None:
        logData = self.LogTransform(data)
        self.mean = torch.mean(logData, dim=0)
        self.std = torch.std(logData, dim=0)
        self.isFitted = True

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        scaledData = (self.LogTransform(data) - self.mean) / (self.std + self.eps)
        return scaledData

    def inverse_transform(self, scaledData: torch.Tensor) -> torch.Tensor:
        logData = scaledData * (self.std + self.eps) + self.mean
        return self.InverseLogTransform(logData)

    def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
        self.fit(data)
        return self.transform(data)
