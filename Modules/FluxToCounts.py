from typing import Callable
import torch

'''
This python module provides functions to reconstruct the detection bin count rates, 
given a differential flux spectrum and detector response functions
'''

def FluxToCounts(fluxSpectrum: torch.Tensor, responseFunctions: torch.Tensor, energies: torch.Tensor) -> torch.Tensor:
    nDims = len(fluxSpectrum.shape)
    integrand = fluxSpectrum.unsqueeze(nDims-1) * responseFunctions
    return torch.trapezoid(integrand, energies, dim=-1)


def FluxParametersToCounts(fluxSpectrum: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                           fluxParameters: torch.Tensor,
                           responseFunctions: torch.Tensor,
                           energies: torch.Tensor) -> torch.Tensor:
    fluxSpectrum = fluxSpectrum(energies, fluxParameters)
    return FluxToCounts(fluxSpectrum, responseFunctions, energies)
