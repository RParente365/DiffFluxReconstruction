import torch
from torch import nn

'''
Neural Network template class with variable inputDim, outputDim and layerWidths
for all the neural networks in the ensembles
'''
class NeuralNetwork(nn.Module):
    def __init__(self, inputDim, outputDim, layerWidths):
        super(NeuralNetwork, self).__init__()
        self.inputLayer = nn.Sequential(
            nn.Linear(inputDim, layerWidths[0]),
            nn.ReLU()
        )
        # List comprehension, followed by star expression unpacking
        self.hiddenLayers = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(layerWidths[i], layerWidths[i+1]),
                nn.ReLU()
            ) for i in range(len(layerWidths)-1)
        ])
        self.outputLayer = nn.Linear(layerWidths[-1], outputDim)

    def forward(self, x) -> torch.Tensor:
        x = self.inputLayer(x)
        x = self.hiddenLayers(x)
        x = self.outputLayer(x)
        return x


# Define the model's initial weights to avoid exploding/vanishing gradients at the start of the training
def init_weights_kaiming(module: nn.Module) -> None:
    for subModule in module.modules():
        if isinstance(subModule, nn.Linear):
            nn.init.kaiming_normal_(subModule.weight, nonlinearity="relu")
            if subModule.bias is not None:
                # Initialize biases to zero
                nn.init.zeros_(subModule.bias)

