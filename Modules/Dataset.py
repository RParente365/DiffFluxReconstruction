import torch
from Modules import LogStandardScaler

'''
Dataset class (that inherits from torch.utils.data.Dataset) needed by the 
DataLoader class (torch.utils.data.DataLoader) to load batches of data samples during training 

This class also implements LogStandardization, useful for input preprocessing
'''

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: torch.Tensor, nTargets: int):
        self.rawData = data
        self.scaledData = data.clone()
        self.nTargets = nTargets

    def PreprocessData(self, featureScaler: LogStandardScaler, targetScaler: LogStandardScaler, NormalizeFeatures: bool = True):
        if NormalizeFeatures:
            self.scaledData[:, self.nTargets:] = self.rawData[:, self.nTargets:] / torch.sum(self.rawData[:, self.nTargets:], dim=-1).unsqueeze(-1)

        if featureScaler.isFitted:
            self.scaledData[:, self.nTargets:] = featureScaler.transform(self.scaledData[:, self.nTargets:])
        else:
            self.scaledData[:, self.nTargets:] = featureScaler.fit_transform(self.scaledData[:, self.nTargets:])

        if self.nTargets > 0:
            if targetScaler.isFitted:
                self.scaledData[:, :self.nTargets] = targetScaler.transform(self.rawData[:, :self.nTargets])
            else:
                self.scaledData[:, :self.nTargets] = targetScaler.fit_transform(self.rawData[:, :self.nTargets])

    def __len__(self):
        return len(self.scaledData)

    def __getitem__(self, idx):
        features = self.scaledData[idx, self.nTargets:]
        targets = self.scaledData[idx, :self.nTargets]
        return features, targets
