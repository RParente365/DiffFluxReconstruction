import torch
from torch.utils.data import random_split
import pickle
from time import time
from Modules import ResponseFunctionsLoader, FluxToCounts, GetRootPath, LogStandardScaler, Dataset

'''
# This script simulates the number of counts "measured" by RADEM for multiple combinations
# of flux spectrum parameters using the RADEM's Response Functions
'''

start = time()

# Defining the flux spectrum parameter tensors
nPars = 2
nSpectralIndices = 500
nRolloverEnergies = 50

# Spectral Indices in [0.5, 5]
SpectralIndices = torch.linspace(0.5, 5, nSpectralIndices)
# Rollover Energy in [1, 100] MeV
RolloverEnergies = torch.logspace(0, 2, nRolloverEnergies)

# Choosing what detection bins will be simulated
detectionBinList = [1, 3, 4, 5, 6]

# Reading RADEM's response functions
responseFunctionsLoader = ResponseFunctionsLoader()
responseFunctionsLoader.LoadResponseFunctions(detectionBinList=detectionBinList)
responseFunctions = responseFunctionsLoader.tensorRF
energies = responseFunctionsLoader.energies

grids = torch.meshgrid(SpectralIndices, RolloverEnergies, indexing="ij")
SpectralIndices_grid = grids[0].unsqueeze(-1)
RolloverEnergies_grid = grids[1].unsqueeze(-1)

# Calculate the flux spectra for all combinations of Spectral Indices and Rollover Energies
FluxSpectra = torch.pow(energies, -SpectralIndices_grid) * torch.exp(-energies/RolloverEnergies_grid)

# Integrate the flux spectra, then multiply by the intensities to get the simulated counts
Counts = FluxToCounts(FluxSpectra, responseFunctions, energies)

# After integrating, the counts are concatenated to the grid tensors 
simData = torch.concat((SpectralIndices_grid, RolloverEnergies_grid, Counts), dim=-1)

# Create the LogStandardScaler scalers for features and targets, to be used when working with the neural networks
simData = torch.flatten(simData, 0, nPars-1)
print(simData.shape)
featureScaler, targetScaler = LogStandardScaler(), LogStandardScaler()

# Create a dataset with the simulated data, then split it into training, validation and test datasets
simDataset = Dataset(simData, nTargets=nPars)
simDataset.PreprocessData(featureScaler, targetScaler)
generator = torch.Generator().manual_seed(365)
simDatasetList = random_split(simDataset, [0.6, 0.2, 0.2], generator)

# Store the dataset in a file
# Open file using "with" to ensure the file is closed properly
with open(f"{GetRootPath()}/Data/NeuralNetworks/DatasetList_RPLS.pkl", "wb") as filehandler:
    pickle.dump(simDatasetList, filehandler)

# Store the LogStandardScalers in a file
# Open file using "with" to ensure the file is closed properly
logStandardScalers = (featureScaler, targetScaler)
with open(f"{GetRootPath()}/Data/NeuralNetworks/LogStandardScalers_RPLS.pkl", "wb") as filehandler:
    pickle.dump(logStandardScalers, filehandler)

print(f"{(time() - start):.3f} s")
