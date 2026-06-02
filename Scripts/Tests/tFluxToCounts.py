from Modules import FluxToCounts, FluxParametersToCounts, ResponseFunctionsLoader
import time
import torch

'''
This python script is used to test the functions in the FluxToCounts module
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

maxEnergy = 300
p0 = 3e4
p1 = 3

nDetectionBins = 8
detectionBinList = [i+1 for i in range(nDetectionBins)]
responseFunctionsLoader = ResponseFunctionsLoader()
responseFunctionsLoader.LoadResponseFunctions(detectionBinList=detectionBinList)
responseFunctionsLoader.SetTensorsToDevice(device)
responseFunctions = responseFunctionsLoader.tensorRF
energies = responseFunctionsLoader.energies

FluxSpectrum = p0*torch.pow(energies, -p1)
FluxSpectrum = FluxSpectrum.to(device)

FluxParameters = torch.tensor([p0, p1], dtype=torch.float32).to(device)
FluxSpectrumFunc = lambda Energies, Parameters: Parameters[0].unsqueeze(-1)*torch.pow(Energies, -Parameters[1].unsqueeze(-1))

startTime = time.time()
print(FluxToCounts(FluxSpectrum, responseFunctions, energies))
print(time.time()-startTime, "seconds")

startTime = time.time()
print(FluxParametersToCounts(FluxSpectrumFunc, FluxParameters, responseFunctions, energies))
print(time.time()-startTime, "seconds")
