import pandas as pd
import numpy as np
import torch
from typing import List
from Modules.RootPath import GetRootPath

'''
This python module provides the class ResponseFunctionLoader, which reads the response functions
of the RADEM detection bins from a folder and stores it in both a Pandas dataframe and PyTorch tensor
'''

class ResponseFunctionsLoader:
    def __init__(self):
        self.dataFrameRF = pd.DataFrame()
        self.detectionBinList = []
        self.tensorRF: torch.Tensor = torch.empty(0)
        self.energies: torch.Tensor = torch.empty(0)

    def SetTensorsToDevice(self, device: torch.device) -> None:
        self.energies = self.energies.to(device)
        self.tensorRF = self.tensorRF.to(device)

    # Method to load the chosen response functions from files to a pandas DataFrame and a PyTorch Tensor
    # detectionBinList can only contain numbers from 1 to 8 (EDH and PDH both have 8 detection bins)
    # folderPath is relative to the project root directory
    # Possible detectorHeads: "PDH", "EDH"
    # Possible particles: "Proton", "Electron"
    def LoadResponseFunctions(self,
                              detectionBinList: List[int],
                              folderPath: str = "Data/ResponseFunctions",
                              detectorHead: str = "PDH",
                              particle: str = "Proton") -> None:
        if len(detectionBinList) == 0:
            raise ValueError("detectionBinList can't be empty: ResponseFunctionLoader")

        self.detectionBinList = detectionBinList
        particleDirectory = None
        if particle == "Proton":
            particleDirectory = "ProtonResponseFunctions"
        elif particle == "Electron":
            particleDirectory = "ElectronResponseFunctions"
        else:
            raise ValueError(f"{particle} isn't a valid particle: ResponseFunctionsLoader.LoadResponseFunctions")

        columnList = []
        columnDataList = []
        for i, detectionBin in enumerate(detectionBinList):
            detectionBinName = None
            if detectorHead == "PDH":
                detectionBinName = f"PROTONS{detectionBin}"
            elif detectorHead == "EDH":
                detectionBinName = f"ELECTRONS{detectionBin}"
            else:
                raise ValueError(f"{detectorHead} isn't a valid detectorHead: ResponseFunctionsLoader.LoadResponseFunctions")

            filename = f"{GetRootPath()}/{folderPath}/{particleDirectory}/Configuration_1/{detectionBinName}.csv"
            df_ = pd.read_csv(filename, sep=";")
            df_["GF"] = df_["GF"].replace(np.nan, 0)
            df_ = df_[df_["Energy"] > 0]
            if i == 0:
                self.energies = torch.tensor(df_["Energy"].values, dtype=torch.float32)
            # Remove NaN values
            columnList.append(f"{detectorHead}{detectionBin}")
            columnDataList.append(df_["GF"].to_numpy(dtype=np.float32))

        columnDataArray = np.array(columnDataList).T
        self.dataFrameRF = pd.DataFrame(data=columnDataArray, columns=columnList)
        self.tensorRF = torch.tensor(self.dataFrameRF.to_numpy().T, dtype=torch.float32)

