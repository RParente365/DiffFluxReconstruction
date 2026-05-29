import numpy as np
import time
import torch
import pickle
from torch import nn
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from Modules import GetRootPath, NeuralNetwork, init_weights_kaiming

# Define the neural network hyperparameters
batchSize = 64
# List of dimensions of the layers
layerWidths = [512, 64, 1024, 512]
initialLearningRate = 2e-4
nEpochs = 500

trainLossesDict = {}
keys = [
    "3DetectionBins",
    "4DetectionBins",
    "5DetectionBins"
]

# Start measuring the time taken to train the ensembles
trainingTimeStart = time.time()
# Loop over the ensembles
for key in keys:
    # Load the logStandardScalers for features and targets
    with open(f"{GetRootPath()}/Data/NeuralNetworks/LogStandardScalers_RPLS_VariableInputs/{key}.pkl", "rb") as filehandler:
        logStandardScalers = pickle.load(filehandler)
    featureScaler, targetScaler = logStandardScalers

    # Load the list of simulated datasets
    with open(f"{GetRootPath()}/Data/NeuralNetworks/DatasetList_RPLS_VariableInputs/{key}.pkl", "rb") as filehandler:
        simDatasetList = pickle.load(filehandler)
    nTargets = simDatasetList[0].dataset.nTargets
    nFeatures = simDatasetList[0].dataset[0][0].shape[0]
    # Initialize the training batch loader
    trainDataLoader = DataLoader(simDatasetList[0], batch_size=batchSize, shuffle=True)
    print(f"trainDataLoader has {len(trainDataLoader)} batches of size {batchSize}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    nNeuralNetworks = 20
    # Loop over the neural networks of the ensemble
    for i in range(nNeuralNetworks):
        # Initialize the neural network
        neuralNetwork = NeuralNetwork(inputDim=nFeatures, outputDim=nTargets, layerWidths=layerWidths).to(device)
        neuralNetwork.apply(init_weights_kaiming)

        if i == 0:
            print(neuralNetwork)
            print("nParameters =", sum(p.numel() for p in neuralNetwork.parameters() if p.requires_grad))

        print("*********************************************************************")
        print(f"Starting the {key} Neural Network Training Loop ({i+1}/{nNeuralNetworks})")
        print("*********************************************************************")

        # Define the costFunction for training
        costFunction = nn.MSELoss()

        # Define the optimizer and the learning rate scheduler
        optimizer = torch.optim.Adam(neuralNetwork.parameters(), lr=initialLearningRate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=nEpochs)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        trainLosses = []
        for epoch in range(nEpochs):
            # Training phase
            neuralNetwork.train()
            trainLoss = 0
            for features, targets in trainDataLoader:
                features, targets = features.to(device), targets.to(device)
                optimizer.zero_grad()
                predParameters = neuralNetwork(features)
                loss = costFunction(predParameters, targets)
                loss.backward()
                optimizer.step()
                trainLoss += torch.sqrt(loss).item()

            trainLosses.append(trainLoss/len(trainDataLoader))
            scheduler.step()

            # epochStep defines how often the losses are printed
            epochStep = 1
            if (epoch+1) % epochStep == 0:
                print("---------------------------------------------------------------------")
                print(f"Epoch {epoch+1}: Training Loss = {trainLosses[epoch]:.7f}")
                print(f"Learning Rate = {scheduler.get_last_lr()[0]}")
                print("---------------------------------------------------------------------")

        trainLossesDict[key] = trainLosses
        # Save the parameters of the trained neural network in a .pth file
        torch.save(neuralNetwork.state_dict(), f"{GetRootPath()}/Data/NeuralNetworks/NeuralNetworkEnsemble_RPLS_VariableInputs/{key}/NeuralNetwork{i+1}.pth")

# Calculate the time taken to train the neural network
trainingTime = time.time() - trainingTimeStart
print(f"Training time = {trainingTime:.3f} s")
