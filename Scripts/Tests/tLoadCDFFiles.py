from Modules import RadMonitorDataLoader, GetRootPath
from matplotlib import pyplot as plt
from datetime import datetime

'''
This python script tests the RadMonitorData class from the RadMonitorData module
'''

keys = ["PROTONS", "ELECTRONS"]
timeIntervalsList = [
    [datetime(2024, 1, 29), datetime(2024, 1, 31)],
    [datetime(2024, 2, 9), datetime(2024, 2, 16)],
    [datetime(2024, 3, 15), datetime(2024, 3, 19)],
    [datetime(2024, 3, 23), datetime(2024, 3, 27)],
    [datetime(2024, 5, 10), datetime(2024, 5, 18)],
    [datetime(2024, 6, 7), datetime(2024, 6, 14)],
    [datetime(2023, 8, 31), datetime(2024, 7, 9)],
    [datetime(2023, 8, 31), datetime(2025, 2, 17)]
]
timeStep = 10 # min

for timeInterval in timeIntervalsList:
    # Initialize a RadMonitorData object and read the data from the cdf files
    radMonitorDataLoader = RadMonitorDataLoader()
    initialDate, finalDate = timeInterval
    radMonitorDataLoader.LoadCDFFiles(keys, initialDate, finalDate)
    for key in keys:
        radMonitorDataFrame = radMonitorDataLoader.dataFrames[key]
        # Rebinning to 1 data point every 10 mins
        radMonitorDataFrame = radMonitorDataFrame.resample(f"{timeStep}min").mean()
        # Converting from counts/min to counts/s
        radMonitorDataFrame = radMonitorDataFrame.div(60)
        plt.figure(figsize=(8, 5), dpi=200)
        colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C9"]
        for i, column in enumerate(radMonitorDataFrame.columns):
            label = f"PDH{column[-1]}" if key == "PROTONS" else f"EDH{column[-1]}"
            plt.plot(radMonitorDataFrame.index, radMonitorDataFrame[column], color=colors[i], label=label)
        plt.yscale("log")
        plt.xticks(rotation=15)
        plt.ylabel(r"Count Rate [s$^{-1}$]")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{GetRootPath()}/Images/RADEM_MeasuredCountRates/{key}:{initialDate.day}-{initialDate.month}-{initialDate.year}_{finalDate.day}-{finalDate.month}-{finalDate.year}.jpg")
        plt.show()
