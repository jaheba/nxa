import numpy as np

from .tensor import Tensor, unify


class TensorFrame:
    def __init__(self, tensors, index=None, shared_dims=None):
        self.tensors = tensors
        self.index = index
        self.shared_dims = shared_dims

        self._verify()

    def _verify(self):
        self.shape = {}
        if self.shared_dims is None:
            return

        for dim in self.shared_dims:
            lengths = set(tensor.shape.get(dim) for tensor in self.tensors.values())
            assert len(lengths) == 1
            length = lengths.pop()
            assert length is not None

            self.shape[dim] = length

    def matches_shape(self, frame):
        for name, length in self.shape.items():
            other = frame.shape.get(name, length)
            if other != length:
                return False
        return True

    def add(self, **frames):
        for name, frame in frames.items():
            assert self.matches_shape(frame)
            self.tensors[name] = frame

    def stack_into(self, target, sources, axis, drop=True):
        unified = unify([self.tensors[source] for source in sources])

        if axis in unified[0].shape:
            result = np.concatenate(unified, axis=axis)
        else:
            result = np.stack(unified, axis=axis)

        if drop:
            for source in sources:
                del self.tensors[source]

        self.tensors[target] = result
        return self
