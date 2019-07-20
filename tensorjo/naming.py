"""This file is responsible for naming nodes and stuff dynamically."""
import numpy as np


def get_node_name(prefix: str = "unknown"):
    """Return a random node name."""
    return prefix + " node_%s" % str(np.random.rand() * 100)


def get_tensor_name():
    """Return a random tensor name."""
    return "tensor_%s" % str(np.random.rand() * 100)
