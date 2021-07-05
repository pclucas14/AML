import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from methods.er import ER

# Abstract Class
class MIR(Method):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # overwrite buffer retrieval function
        self.buffer.add = 1


    def _mir_add(self, buffer):
        pass
