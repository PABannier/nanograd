from enum import Enum

class Device(Enum):
    r"""
        Enumeration of the devices supported by
        Nanograd. 
        Currently, Nanograd only supports CPU and GPU.
    """
    CPU = 1
    GPU = 2
