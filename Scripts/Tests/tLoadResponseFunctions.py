from Modules import ResponseFunctionsLoader, GetRootPath
from matplotlib import pyplot as plt
from itertools import product

'''
This python script tests the RFunctions class from the RFunctions module
'''

detectorHeads = ["PDH", "EDH"]
particles = ["Proton", "Electron"]
options = list(product(detectorHeads, particles))

for option in options:
    nDetectionBins = 8
    detectionBinList = [i+1 for i in range(nDetectionBins)]
    responseFunctionsLoader = ResponseFunctionsLoader()
    responseFunctionsLoader.LoadResponseFunctions(detectionBinList=detectionBinList, detectorHead=option[0], particle=option[1])
    responseFunctions = responseFunctionsLoader.tensorRF
    energies = responseFunctionsLoader.energies
    plt.figure(figsize=(8, 5), dpi=200)
    colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C9"]
    for i in range(responseFunctions.shape[0]):
            plt.plot(energies, responseFunctions[i, :], color=colors[i], label=f"{option[0]}{detectionBinList[i]}")

    if option[1] == "Proton":
        plt.xlim(4, energies[-1])
    elif option[1] == "Electron":
        plt.xlim(0.3, energies[-1])
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Energy [MeV]")
    plt.ylabel(r"Response Function [cm$^{2}$sr]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{GetRootPath()}/Images/ResponseFunctions/{option[0]}_{option[1]}s_Config1.jpg")
    plt.show()
