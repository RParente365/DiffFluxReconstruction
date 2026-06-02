import torch
import pickle
from Modules import GetRootPath, NeuralNetwork, ResponseFunctionsLoader, Dataset, FluxParametersToCounts

'''
DiffFluxReconstructor is a functor that reconstructs the proton differential flux spectrum parameters
from detection bin count rates
'''

class DiffFluxReconstructor:
    def __init__(self):
        # Setting the device used by PyTorch during inference to the gpu if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        ############################################################
        # Loading the trained neural networks of the deep ensembles
        ############################################################
        self.nDetectionBins = 5
        self.nNeuralNetworksPerEnsemble = 20
        self.detectionBinListDict = {
            "3DetectionBins": [1, 3, 4],
            "4DetectionBins": [1, 3, 4, 5],
            "5DetectionBins": [1, 3, 4, 5, 6]
        }
        layerWidths = [512, 64, 1024, 512]
        # Defining the directories from which the neural networks and dataset scalers are loaded
        modelParametersDir = "Data/NeuralNetworks/NeuralNetworkEnsemble_RPLS_VariableInputs"
        datasetScalersDir = "Data/NeuralNetworks/LogStandardScalers_RPLS_VariableInputs"
        self.deepEnsembles = {}
        self.datasetScalers = {}
        for key, detectionBinList in self.detectionBinListDict.items():
            # Looping through each detectionBinList and load a deep ensemble with the corresponding input detection bins
            deepEnsemble = []
            for i in range(self.nNeuralNetworksPerEnsemble):
                # Creating a new neural network instance
                neuralNetwork = NeuralNetwork(inputDim=len(detectionBinList), outputDim=2, layerWidths=layerWidths)
                # Loading the parameters of the corresponding trained neural network in the ensemble
                neuralNetwork.load_state_dict(
                    torch.load(
                        f"{GetRootPath()}/{modelParametersDir}/{key}/NeuralNetwork{i+1}.pth",
                        map_location=self.device
                    )
                )
                # Moving neural network to device for inference
                neuralNetwork.to(self.device)
                # Setting neural network to evaluation mode
                neuralNetwork.eval()
                # Adding the loaded neural network to the ensemble list
                deepEnsemble.append(neuralNetwork)
            # Storing the loaded ensemble list in the deepEnsembles dictionary
            self.deepEnsembles[key] = deepEnsemble
            # Loading the dataset scalers for input preprocessing and output postprocessing
            with open(f"{GetRootPath()}/{datasetScalersDir}/{key}.pkl", "rb") as filehandler:
                self.datasetScalers[key] = pickle.load(filehandler)
        # Loading the response function used for count rate reconstruction to estimate the differential flux intensity
        responseFunctionsLoader = ResponseFunctionsLoader()
        responseFunctionsLoader.LoadResponseFunctions(detectionBinList=[1, 3, 4, 5, 6])
        self.responseFunctions = responseFunctionsLoader.tensorRF
        self.responseFunctionEnergies = responseFunctionsLoader.energies

    # Parametrized function for the differential flux spectrum
    @staticmethod
    def RollingPowerLaw(energies: torch.Tensor, parameters: torch.Tensor):
        return torch.pow(energies, -parameters[:, 0].unsqueeze(-1)) * torch.exp(-energies / parameters[:, 1].unsqueeze(-1)) 

    '''
    Functor call function that estimates the differential flux spectrum parameters from detection bin count rates
    Inputs:
        countRates: Detection bin count rates
        bkgMean: Detection bin background means (ensemble selection)
    Outputs:
        predParameters: Ensemble-averaged differential flux parameters
        predParameterUncertainties: Differential flux parameter ensemble uncertainties
        predParameterSamples: Diffferential flux parameter ensemble outputs and estimated differential flux intensities (non-averaged)
        maskDict: Mask dictionary from ensemble selection (useful to know which ensemble was used to estimate the differential flux parameters)
    '''
    def __call__(self, countRates: torch.Tensor, **kwargs):
        # If no bkgMean is given assume, it is 0
        bkgMean = kwargs.get("bkgMean", torch.zeros(self.nDetectionBins))

        ###############################
        # Ensemble Selection
        ###############################
        # Creating a mask dictionary used for ensemble selection
        maskDict = {} # key -> nDetectionBins, value -> mask
        maskDict["3DetectionBins"] = countRates[:, 3] < (2 * bkgMean[3])
        maskDict["4DetectionBins"] = (countRates[:, 4] < (2 * bkgMean[4])) & ~maskDict["3DetectionBins"]
        maskDict["5DetectionBins"] = ~maskDict["3DetectionBins"] & ~maskDict["4DetectionBins"]
        # Creating a dictionary of indices that selects the count rates to be process by a given ensemble using the mask dictionary
        indicesDict = {} # key -> nDetectionBinsInputs, value -> mask nonzero indices
        for key, mask in maskDict.items():
            indicesDict[key] = mask.nonzero(as_tuple=True)[0]

        # Creating the output tensors to be filled after inference
        predParameters = torch.zeros(countRates.shape[0], 3)
        predParameterUncertainties = torch.zeros(countRates.shape[0], 3)
        predParameterSamples = torch.zeros(countRates.shape[0], 3, self.nNeuralNetworksPerEnsemble)
        # Loop over the ensembles to process the count rates to them during ensemble selection
        for key, mask in maskDict.items():
            # If this ensemble was not chosen to process any count rates, move on to the next ensemble
            if mask.count_nonzero() == 0:
                continue
            # Preprocess the input count rates to be processed by this ensemble
            nDetectionBins = len(self.detectionBinListDict[key])
            countRatesDataset = Dataset(countRates[mask, :nDetectionBins], nTargets=0)
            featureScaler, targetScaler = self.datasetScalers[key]
            countRatesDataset.PreprocessData(featureScaler, targetScaler)
            preprocessedCountRates = countRatesDataset[:][0].to(self.device)

            # The preprocessed inputs are passed to the neural networks in the ensemble, and their outputs are postprocessed
            # by the inverse_transform method of the targetScaler object
            predParameterSamples_t = []
            with torch.no_grad():
                for neuralNetwork in self.deepEnsembles[key]:
                    predParameterSamples_t.append(targetScaler.inverse_transform(neuralNetwork(preprocessedCountRates).cpu()))
            predParameterSamples_t = torch.stack(predParameterSamples_t, dim=-1)
            # Averaging over the ensemble
            predParameters_t = torch.mean(predParameterSamples_t, dim=-1)
            # Calculating the ensemble uncertainty 
            predParameterUncertainties_t = torch.std(predParameterSamples_t, dim=-1)

            # Estimate differential flux intensity with linear regression
            predIntensitySamples = []
            for i in range(self.nNeuralNetworksPerEnsemble):
                reconstructedCountRates = FluxParametersToCounts(self.RollingPowerLaw, predParameterSamples_t[:, :, i], self.responseFunctions, self.responseFunctionEnergies)
                X = reconstructedCountRates[:, :nDetectionBins].unsqueeze(-1)
                Y = countRatesDataset.rawData.unsqueeze(-1)
                predIntensitySample = torch.bmm(torch.transpose(X, 1, 2), Y).squeeze(-1) / torch.bmm(torch.transpose(X, 1, 2), X).squeeze(-1)
                predIntensitySamples.append(predIntensitySample)

            predIntensitySamples = torch.stack(predIntensitySamples, dim=-1)
            # Averaged over the ensemble
            predIntensity = torch.mean(predIntensitySamples, dim=-1)
            # Estimate the ensemble uncertainty of the intensity
            predIntensityUncertainties = torch.std(predIntensitySamples, dim=-1)
            # Concatenate the parameters estimated by the neural networks with the estimated intensity
            predParameters_t = torch.cat((predIntensity, predParameters_t), dim=-1)
            predParameterUncertainties_t = torch.cat((predIntensityUncertainties, predParameterUncertainties_t), dim=-1)
            predParameterSamples_t = torch.cat((predIntensitySamples, predParameterSamples_t), dim=1)

            # Store the estimated differential flux parameters in the corresponding indices of the output tensors
            predParameters[indicesDict[key]] = predParameters_t
            predParameterUncertainties[indicesDict[key]] = predParameterUncertainties_t
            predParameterSamples[indicesDict[key]] = predParameterSamples_t
        return predParameters, predParameterUncertainties, predParameterSamples, maskDict


