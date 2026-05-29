from spacepy import pycdf
from datetime import datetime
from Modules import GetRootPath
\
'''
This python script is used to test reading CDF files, 
and to see what keys are available
'''

Date = datetime(2023, 8, 31)
dateStr = Date.strftime("%Y%m%d")
filepath = f"{GetRootPath()}/Data/FullData-RAW/pds/radem_raw_sc_{dateStr}__0_1.cdf"
with pycdf.CDF(filepath) as cdffile:
    print(cdffile.keys())
    varInspect = "PROTONS"
    print(cdffile[varInspect].shape)
    # CDF object behave like a library
    if varInspect in cdffile:
        print(f"{varInspect} is in {filepath}")
    else:
        print(f"{varInspect} is not in {filepath}")
