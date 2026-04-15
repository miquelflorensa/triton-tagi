"""TAGI layers: building blocks for Bayesian neural networks."""

from .avgpool2d import AvgPool2D
from .batchnorm2d import BatchNorm2D
from .bernoulli import Bernoulli
from .conv2d import Conv2D
from .even_softplus import EvenSoftplus
from .flatten import Flatten
from .frn import FRN2D
from .frn_resblock import FRNResBlock
from .leaky_relu import LeakyReLU
from .linear import Linear
from .relu import ReLU
from .remax import Remax
from .resblock import Add, ResBlock
from .shared_var_batchnorm2d import SharedVarBatchNorm2D
from .shared_var_conv2d import SharedVarConv2D
from .shared_var_linear import SharedVarLinear
from .shared_var_resblock import SharedVarResBlock
from .silu import SiLU
from .tlu import TLU

__all__ = [
    "AvgPool2D",
    "BatchNorm2D",
    "Bernoulli",
    "Conv2D",
    "EvenSoftplus",
    "Flatten",
    "FRN2D",
    "FRNResBlock",
    "LeakyReLU",
    "Linear",
    "ReLU",
    "Remax",
    "Add",
    "ResBlock",
    "SharedVarBatchNorm2D",
    "SharedVarConv2D",
    "SharedVarLinear",
    "SharedVarResBlock",
    "SiLU",
    "TLU",
]
