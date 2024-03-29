"""Base object."""
from . import tensor as tensor_base
from . import naming
from . import math
from . import graph
from . import node

from tensorjo import ops
from tensorjo import opt

tensor = tensor_base.tensor

monoid = node.monoid
functor = node.functor
primitive = node.primitive


ops = ops
opt = opt

naming = naming

add = math.add
sub = math.sub
div = math.div
mul = math.mul
mse = math.mse
var = math.var
gradients = math.gradients

sigmoid = math.sigmoid
sin = math.sin
cos = math.cos

tjgraph = graph.graph("default")
