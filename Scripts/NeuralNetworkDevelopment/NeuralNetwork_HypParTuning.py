from functools import partial
import tempfile
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from ray import tune
from ray.tune import Tuner, TuneConfig, RunConfig, CheckpointConfig, Checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from typing import Optional, Tuple, Dict, Any
from torch.utils.data import Subset
import ray.cloudpickle as cloudpickle
from optuna.samplers import TPESampler
from Modules import GetRootPath, NeuralNetwork, init_weights_kaiming

'''
This script is used to tune the hyperparameters of the neural networks of the machine learning model
The hyperparameter optimization is done for a single neural network, and the results are used
on all the neural networks of the ensembles
'''

# This function is used to load the training, validation and test datasets
def LoadDatasets(datasetsPath) -> Tuple[Subset, Subset, Subset]:
    with open(datasetsPath, "rb") as filehandler:
        simDatasetList = cloudpickle.load(filehandler)
    # return trainDataset, valDataset, testDataset
    return simDatasetList[0], simDatasetList[1], simDatasetList[2]


# This function is used by Ray Tune to train different configs of the NeuralNetwork
def Train(config, maxEpochs, datasetsPath) -> None:
    # Initialize the neuralNetwork with the given config
    neuralNetwork = NeuralNetwork(
        inputDim=5,
        outputDim=2,
        layerWidths=[config[f"layerWidth{i+1}"] for i in range(config["nLayers"])]
    )
    neuralNetwork.apply(init_weights_kaiming)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        if torch.cuda.device_count() > 1:
            neuralNetwork = nn.DataParallel(neuralNetwork)
    neuralNetwork.to(device)

    # Initialize the costFunction, optimizer and learning rate scheduler
    costFunction = nn.MSELoss()
    optimizer = torch.optim.Adam(neuralNetwork.parameters(), lr=config["initialLearningRate"])
    lrScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=maxEpochs)

    # If there is a NeuralNetwork config that hasn't finished training, continue training it
    checkpoint = tune.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpointDir:
            dataPath = Path(checkpointDir) / "CheckpointData.pkl"
            with open(dataPath, "rb") as filehandler:
                checkpointState = cloudpickle.load(filehandler)
            startEpoch = checkpointState["epoch"]
            neuralNetwork.load_state_dict(checkpointState["neuralNetwork_state_dict"])
            optimizer.load_state_dict(checkpointState["optimizer_state_dict"])
            lrScheduler.load_state_dict(checkpointState["lrScheduler_state_dict"])
    # Else, start from scratch
    else:
        startEpoch = 1

    batchSize = 64
    trainDataset, valDataset, testDataset = LoadDatasets(datasetsPath)
    trainDataLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
    valDataLoader = DataLoader(valDataset, batch_size=batchSize)
    for epoch in range(startEpoch, maxEpochs+1):
        # Training phase
        neuralNetwork.train()
        trainLoss = 0
        trainSteps = 0
        for features, targets in trainDataLoader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            predParameters = neuralNetwork(features)
            loss = costFunction(predParameters, targets)
            loss.backward()
            optimizer.step()
            trainLoss += torch.sqrt(loss).item()
            trainSteps += 1

        # Validation phase
        neuralNetwork.eval()
        valLoss = 0
        valSteps = 0
        with torch.no_grad():
            for features, targets in valDataLoader:
                features, targets = features.to(device), targets.to(device)
                predParameters = neuralNetwork(features)
                loss = costFunction(predParameters, targets)
                valLoss += torch.sqrt(loss).item()
                valSteps += 1

        lrScheduler.step()

        # Report the valLoss of this trained NeuralNetwork configuration to Ray Tune
        checkpointData = {
            "epoch": epoch,
            "neuralNetwork_state_dict": neuralNetwork.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lrScheduler_state_dict": lrScheduler.state_dict()
        }
        with tempfile.TemporaryDirectory() as checkpointDir:
            dataPath = Path(checkpointDir) / "CheckpointData.pkl"
            with open(dataPath, "wb") as filehandler:
                cloudpickle.dump(checkpointData, filehandler) # pyright: ignore

            checkpoint = Checkpoint.from_directory(checkpointDir)
            tune.report(
                {"loss": valLoss / valSteps},
                checkpoint=checkpoint,
            )


# This function is used to test the NeuralNetwork configuration with the lowest valLoss
def Test(neuralNetwork, device, datasetsPath) -> float:
    batchSize = 400
    trainDataset, valDataset, testDataset = LoadDatasets(datasetsPath)
    testDataLoader = DataLoader(testDataset, batch_size=batchSize)

    MSELoss = nn.MSELoss()
    def errorFunction(predParameters, targets):
        return torch.sqrt(MSELoss(predParameters, targets))
    testLoss = 0
    with torch.no_grad():
        for features, targets in testDataLoader:
            features, targets = features.to(device), targets.to(device)
            predParameters = neuralNetwork(features)
            testLoss += errorFunction(predParameters, targets).item()

    return testLoss / len(testDataLoader)

# This function defines the hyperparameter space from which the hyperparameter sets are sampled
def defineParamSpace(trial) -> Optional[Dict[str, Any]]:
    # Variable number of layers
    nLayers = trial.suggest_int("nLayers", 1, 4)
    # Variable number of neurons in the number of layers (nLayers) sampled above
    for i in range(nLayers):
        trial.suggest_categorical(f"layerWidth{i+1}", [2**(j+5) for j in range(6)])
    # Variable initial learning rate of the optimizer
    trial.suggest_float("initialLearningRate", 1e-4, 1e-1, log=True)

# Main function
def main(experimentName, nValSamples, maxEpochs, nCheckpointsKept) -> None:
    #######################
    # Setting up the tuner
    #######################
    datasetsPath = f"{GetRootPath()}/Data/NeuralNetworks/DatasetList_RPLS.pkl"
    storageDir = f"{GetRootPath()}/Data/NeuralNetworks/HypParTuningExperiments"
    experimentPath = str(Path(storageDir) / experimentName)
    minEpochs = 20
    # Defining the function used to train the neural network of a given trial, and the computational resources it uses
    trainable = tune.with_resources(
        partial(Train, maxEpochs=maxEpochs, datasetsPath=datasetsPath),
        {"cpu": 2, "gpu": 0.2}
    )
    # Defining the trial scheduler
    scheduler = ASHAScheduler(
        max_t=maxEpochs,
        grace_period=minEpochs,
        reduction_factor=2
    )
    # Defining the hyperparameter search algorithm
    searchAlgo = OptunaSearch(
        space=defineParamSpace,
        metric="loss",
        mode="min",
        sampler=TPESampler()
    )
    # Setting the configurations of the tuner
    tuneConfig = TuneConfig(
        mode="min",
        metric="loss",
        search_alg=searchAlgo,
        scheduler=scheduler,
        num_samples=nValSamples
    )
    checkpointConfig = CheckpointConfig(
        num_to_keep=nCheckpointsKept,
    )
    runConfig = RunConfig(
        name=experimentName,
        storage_path=storageDir,
        checkpoint_config=checkpointConfig,
    )
    # If a hyperparameter tuning run has been interrupted, continue it
    if Tuner.can_restore(experimentPath):
        tuner = Tuner.restore(
            experimentPath,
            trainable=trainable,
            resume_unfinished=True,
            resume_errored=True
        )
    # else, start a new hyperparameter tuning run
    else:
        tuner = Tuner(
            trainable=trainable,
            tune_config=tuneConfig,
            run_config=runConfig,
        )
    # Running the tuner
    results = tuner.fit()

    #########################
    # Presenting the results
    #########################
    bestTrial = results.get_best_result(metric="loss", mode="min", scope="last")
    if bestTrial is None:
        raise RuntimeError("No best trial found")
    print(f"Best trial config: {bestTrial.config}")
    print(f"Best trial final validation loss: {bestTrial.metrics['loss']}") # pyright: ignore
    bestTrainedNN = NeuralNetwork(
        inputDim=5,
        outputDim=2,
        layerWidths=[bestTrial.config[f"layerWidth{i+1}"] for i in range(bestTrial.config["nLayers"])] # pyright: ignore
    )
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        if torch.cuda.device_count() > 1:
            bestTrainedNN = nn.DataParallel(bestTrainedNN)
    bestTrainedNN.to(device)

    bestCheckpoint = bestTrial.get_best_checkpoint(metric="loss", mode="min")
    if bestCheckpoint is None:
        raise RuntimeError("No best checkpoint found")
    with bestCheckpoint.as_directory() as checkpointDir:
        dataPath = Path(checkpointDir) / "CheckpointData.pkl"
        with open(dataPath, "rb") as filehandler:
            bestCheckpointData = cloudpickle.load(filehandler)
        bestTrainedNN.load_state_dict(bestCheckpointData["neuralNetwork_state_dict"])
        testLoss = Test(bestTrainedNN, device, datasetsPath)
        print(f"Best trial test error: {testLoss}")


# If this script is ran, call the main function with the given parameters
if __name__ == "__main__":
    main("RollingPowerLawSim", nValSamples=1000, maxEpochs=500, nCheckpointsKept=5)
