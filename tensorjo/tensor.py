"""Numpy arrays are the base arrays for the tensors."""
import numpy as np
import tensorjo
from . import node


def tensor(v, name: str = None):
    """Convert thing to okay tensor format."""
    if isinstance(v, node.node):
        return v

    try:
        v = np.array(v, dtype=np.float32)
    except Exception as e:
        raise ValueError("Unable to convert %s to float array" % v)

    if len(v.shape) > 0 and v.shape[0] == 0:
        raise ValueError("Empty tensor is not allowed.")

    if np.isnan(v).any():
        raise ValueError("Invalid tensor -- Contains NaN or None.")

    if name is None:
        name = tensorjo.naming.get_tensor_name()

    return node.primitive(v, name)
