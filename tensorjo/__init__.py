"""Base object."""
from . import tensor as tensor_base
from . import naming
from . import math
from . import graph

from tensorjo import ops

tensor = tensor_base.tensor
ops = ops
naming = naming

add = math.add
sub = math.sub
div = math.div
mul = math.mul
mse = math.mse
dot = math.dot
var = math.var
gradients = math.gradients

sigmoid = math.sigmoid

tjgraph = graph.graph("default")
