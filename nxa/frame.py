import numpy as np

from .tensor import Tensor, unify, Ax


class TensorFrame:
    def __init__(self, tensors, index=None, *, dims, props=None):
        self.tensors = tensors
        self.index = index or {}
        self.dims = dims

        self.props = props or {}

        self._verify()

        self.ax = Ax(self)

    def _verify(self):
        self.shape = {}
        if self.dims is None:
            return

        for dim in self.dims:
            lengths = set(tensor.shape.get(dim) for tensor in self.tensors.values())
            assert len(lengths) == 1
            length = lengths.pop()
            assert length is not None

            idx = self.index.get(dim)
            if idx is not None:
                assert len(idx) == length

            self.shape[dim] = length

    def __repr__(self):
        shape = ", ".join(f"{dim}={n}" for dim, n in self.shape.items())
        return f"TensorFrame<{shape}>"

    def __getitem__(self, name):
        return self.tensors[name]

    def slice(self, axis, arg):
        assert axis in self.dims

        tensors = {
            name: tensor.slice(axis, arg) for name, tensor in self.tensors.items()
        }

        sliced, orig = next(zip(tensors.values(), self.tensors.values()))

        if sliced.ndim == orig.ndim:
            dims = self.dims
        else:
            dims = [dim for dim in self.dims if dim != axis]

        index = dict(self.index)

        if axis in index:
            index[axis] = index[axis][arg]

        return TensorFrame(
            tensors,
            index=index,
            dims=dims,
            props=self.props,
        )

    def split_at(self, axis, idx):
        lefts = {}
        rights = {}

        for name, tensor in self.tensors.items():
            left, right = tensor.split_at(axis, idx)
            lefts[name] = left
            rights[name] = right

        left_index = dict(self.index)
        right_index = dict(self.index)
        if axis in self.index:
            left_index[axis] = left_index[axis][:idx]
            right_index[axis] = right_index[axis][idx:]

        return TensorFrame(
            lefts, left_index, dims=self.dims, props=self.props
        ), TensorFrame(
            rights,
            right_index,
            dims=self.dims,
            props=self.props,
        )

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

        self.add(**{target: result})
        return self

    def _map_fn(self, fn, *args, **kwargs):
        return {
            name: fn(tensor, *args, **kwargs) for name, tensor in self.tensors.items()
        }

    def sum(self, axis):
        assert axis in self.shape

        index = dict(self.index)
        index.pop(axis, None)

        return TensorFrame(
            self._map_fn(np.sum, axis=axis),
            index,
            dims=[dim for dim in self.dims if dim != axis],
            props=self.props,
        )
