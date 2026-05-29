from pathlib import Path

'''
This module provides a function that returns the head directory of the project
'''

def GetRootPath() -> Path:
    return Path(__file__).absolute().parent.parent
