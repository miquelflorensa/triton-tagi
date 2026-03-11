"""TAGI layers: building blocks for Bayesian neural networks."""

from .linear import Linear
from .relu import ReLU
from .leaky_relu import LeakyReLU
from .remax import Remax
from .bernoulli import Bernoulli
from .conv2d import Conv2D
from .avgpool2d import AvgPool2D
from .batchnorm2d import BatchNorm2D
from .flatten import Flatten
from .resblock import ResBlock, Add
from .even_softplus import EvenSoftplus

# Shared-variance variants
from .shared_var_linear import SharedVarLinear
from .shared_var_conv2d import SharedVarConv2D
from .shared_var_batchnorm2d import SharedVarBatchNorm2D
from .shared_var_resblock import SharedVarResBlock
