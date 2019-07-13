"""Numpy arrays are the base arrays for the tensors."""
import numpy as np
from . import outputable


class tensor():
    """Tensor object to keep track of operations for a variable."""

    valid_types = [np.float16, np.float32, np.float64, np.float128]

    def __init__(self, v):
        """Initialize tensor by converting the value to a float numpy array."""
        if type(v) == np.ndarray and v.dtype in tensor.valid_types:
            self.v = v

        invalid_type_err = lambda s: ("Invalid tensor type %s" % s)\
            + "Please convert it into a float or float array"

        if type(v) == bytes:
            raise ValueError(invalid_type_err(v))

        if v is None:
            raise ValueError(invalid_type_err(v))

        if hasattr(v, '__len__') and len(v) == 0:
            raise ValueError(invalid_type_err(v))

        try:
            self.v = np.array(v, dtype=np.float32)
        except Exception as e:
            raise ValueError(invalid_type_err(v))

    @property
    def shape(self):
        """Return the shape of the tensor."""
        return self.v.shape

    def __str__(self):
        """Show the underlying array."""
        return str(self.v)
