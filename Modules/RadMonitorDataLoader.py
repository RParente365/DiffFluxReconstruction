import os
from spacepy import pycdf
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict
from Modules.RootPath import GetRootPath

'''
This python module provides the class RadMonitorData, which reads RADEM's data from a folder
and stores it in a pandas DataFrame
'''

class RadMonitorDataLoader:
    # Class constructor
    def __init__(self):
        self.dataFrames: Dict[str, pd.DataFrame] = {}

    # Helper method that generates a filepath for a given datetime
    # folderPath is relative to the project root directory
    def _GenerateFilePath(self, date: datetime, number: int, folderPath: str) -> str:
        dateStr = date.strftime("%Y%m%d")
        return f"{GetRootPath()}/{folderPath}/pds/radem_raw_sc_{dateStr}__0_{number}.cdf"

    # Method that loading data from the cdf files in a given time interval
    # folderPath is relative to the project root directory
    def LoadCDFFiles(self, keys: list[str], initialDate: datetime, finalDate: datetime, folderPath: str = "Data/FullData-RAW") -> None:
        for key in keys:
            time = []
            data = []
            columns = []
            # Loop through all the cdf files in the chosen time interval
            date = initialDate
            while date < finalDate + timedelta(days=1):
                foundFile = False
                # The files can have either 1 or 2 as the last number
                for number in [1, 2]:
                    filePath = self._GenerateFilePath(date, number, folderPath)
                    if os.path.exists(filePath):
                        foundFile = True
                        with pycdf.CDF(filePath) as cdffile:
                            time.extend(pd.to_datetime(cdffile["TIME_UTC"][:]))
                            data.extend(cdffile[key][:])
                            if len(columns) == 0:
                                # Define the names of the pandas DataFrame columns
                                columns.extend([f"{key}_{i+1}" for i in range(cdffile[key].shape[1])])
                        # Exit the loop if a valid file is found
                        break
                if not foundFile:
                    dateStr = date.strftime("%d/%m/%Y")
                    print(f"There is no file for {dateStr}: RadMonitorDataLoader.LoadCDFFiles")

                date += timedelta(days=1)

            newData = pd.DataFrame(data=data, index=time, columns=columns)
            # Only keep data in the given time interval
            newData = newData[newData.index >= pd.Timestamp(initialDate, tz="UTC")]
            newData = newData[newData.index <= pd.Timestamp(finalDate, tz="UTC")]
            # If the given key already has a DataFrame, load the new data into it (non-duplicate)
            if key in self.dataFrames:
                self.dataFrames[key] = pd.concat([self.dataFrames[key], newData], axis=0, ignore_index=False)
            else:
                self.dataFrames[key] = newData
