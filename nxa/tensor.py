from dataclasses import dataclass
from numbers import Number
from typing import Dict

import numpy as np


class Ax:
    def __init__(self, tensor):
        for dimension in tensor.dims:
            setattr(self, dimension, Slicer(tensor, dimension))


@dataclass
class Slicer:
    xs: object
    dim: str

    def __getitem__(self, idx):
        return self.xs.slice(self.dim, idx)


Dims = Dict[str, int]


def equal_dims(a: Dims, b: Dims) -> bool:
    return a == b


def same_dims(a: Dims, b: Dims) -> bool:
    # like `equal_dims`, but keys have same order
    return list(a) == list(b) and a == b


class Tensor(np.lib.mixins.NDArrayOperatorsMixin):
    __slots__ = "ax", "values", "shape"

    _sub_types = {}

    def __class_getitem__(cls, dims):
        if not isinstance(dims, tuple):
            dims = (dims,)

        if dims not in cls._sub_types:
            cls._sub_types[dims] = type(
                f"Tensor[{dims!r}]",
                (Tensor,),
                {
                    "dims": dict(zip(dims, range(len(dims)))),
                },
            )

        return cls._sub_types[dims]

    @classmethod
    def promote(cls, tensor):
        assert set(tensor.shape) <= set(cls.dims)
        return tensor.with_dimensions(list(cls.dims))

    @classmethod
    def without_dims(cls, value, dimensions):
        remaining = cls.dims.keys() - set(dimensions)
        if remaining:
            return Tensor[tuple(remaining)](value)
        return value

    def with_dimensions(self, dimensions):
        other_unique_dims = [dim for dim in dimensions if dim not in self.dims]
        my_unique_dims = [dim for dim in self.dims if dim not in dimensions]

        order = []
        for dim in dimensions:
            try:
                order.append(other_unique_dims.index(dim))
            except ValueError:
                order.append(len(other_unique_dims) + list(self.dims).index(dim))

        order.extend(np.arange(len(order), len(order) + len(my_unique_dims)))
        expanded = np.expand_dims(self.values, tuple(np.arange(len(other_unique_dims))))

        # print(order, expanded.shape, expanded.ndim)
        cls = Tensor[tuple(dimensions + my_unique_dims)]
        return cls(np.transpose(expanded, order))

    @classmethod
    def new(cls, values):
        return cls(values)

    def __new__(cls, *args, **kwargs):
        assert cls is not Tensor
        return object.__new__(cls)

    def __init__(self, values, dtype=None):
        self.values = np.asarray(values, dtype=dtype)
        self.ax = Ax(self)

        assert self.values.ndim == self.ndim
        self.shape = {name: self.values.shape[axis] for name, axis in self.dims.items()}

    @property
    def ndim(self):
        return len(self.__class__.dims)

    @property
    def dtype(self):
        return self.values.dtype

    def unwrap(self, dims=None):
        if dims is None or same_dims(self.dims, dims):
            return self.values

        return np.transpose(self.values, tuple(self.dims[dim] for dim in dims))

    def transpose(self, dims):
        return Tensor[tuple(dims)](self.unwrap(dims))

    def __repr__(self):
        shape = ", ".join(f"{dim}={n}" for dim, n in self.shape.items())
        return f"Tensor<{shape}>"

    def _call(self, ufunc, inputs, kwargs):
        if len(inputs) == 1:
            return self.new(ufunc(self.unwrap(), *kwargs))

        xs = []
        self_idx = None

        for idx, other in enumerate(inputs):
            if other is self:
                self_idx = idx

            if isinstance(other, (Number, Tensor)):
                xs.append(other)
            else:
                return NotImplementedError()

        if len(xs) == 2:
            if all(isinstance(x, Tensor) for x in xs):
                left, right, dims = unwrap_two(xs[0], xs[1])
                return Tensor[tuple(dims)](ufunc(left, right, **kwargs))

            elif self_idx == 0:
                return self.__class__(ufunc(self.unwrap(), xs[1], *kwargs))
            else:
                return self.__class__(ufunc(xs[0], self.unwrap(), *kwargs))

        return self

    def __array_function__(self, func, types, args, kwargs):
        # assert args[0] is self
        if func is np.mean:
            return self.mean(*args[1:], **kwargs)
        elif func is np.stack:
            return self._stack(*args, **kwargs)

        elif func is np.concatenate:
            return self._concatenate(*args, **kwargs)

        raise NotImplementedError(func)

    def mean(self, axis=None, dtype=None):
        if axis is not None:
            idx_axis = self.dims[axis]
        else:
            idx_axis = None

        result = np.mean(self.unwrap(), axis=idx_axis, dtype=dtype)

        return self.without_dims(result, [axis])

    def _stack(self, arrays, axis):
        assert axis not in self.dims
        stacked = np.stack([arr.unwrap(self.dims) for arr in arrays])
        return Tensor[(axis,) + tuple(self.dims)](stacked)

    def _concatenate(self, arrays, axis):
        assert axis in self.dims

        num_axis = list(self.dims).index(axis)

        concatenated = np.concatenate(
            [arr.unwrap(self.dims) for arr in arrays], axis=num_axis
        )
        return self.new(concatenated)

    def __array__(self):
        return self.unwrap()

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):

        if method == "__call__":
            return self._call(ufunc, inputs, kwargs)

        # elif method == "reduce":
        #     axis = kwargs["axis"]
        #     if axis is None:
        #         return ufunc.reduce(self.unwrap(), **kwargs)

        #     kwargs["axis"] = self.dims[axis]
        #     return self.without_dims(axis)(ufunc.reduce(self.values, **kwargs))

        raise NotImplementedError()

    def __getitem__(self, idx):
        if type(idx) == slice and idx == slice(None, None, None):
            return self

        if isinstance(idx, Tensor):
            assert idx.shape.keys() <= self.shape.keys()

            if idx.dtype == bool:
                if idx.shape == self.shape:
                    return self.unwrap()[idx.unwrap(self.dims)]

            current = self
            for dim, dim_idx in idx.shape.items():
                current = current.slice(dim, dim_idx)

            return current

    def __setitem__(self, idx, value):
        if isinstance(idx, Tensor) and idx.dtype == bool:
            if self.shape == idx.shape:
                self.values[idx.unwrap(self.dims)] = value
                return

        raise NotImplementedError

    def slice(self, axis, arg):
        idx = [slice(None)] * len(self.dims)
        idx[self.dims[axis]] = arg
        idx = tuple(idx)
        values = self.values[idx]

        if values.ndim == self.ndim:
            return self.new(values)

        return self.without_dims(values, [axis])

    def select(self, query):
        pass


def unwrap_two(a, b):
    if len(a.shape) > len(b.shape):
        big = a
        small = b
    else:
        big = b
        small = a

    additional = small.shape.keys() - big.shape.keys()
    assert not additional

    order = list(big.shape.keys() - small.shape.keys()) + list(small.shape.keys())

    if a is big:
        return a.unwrap(order), b.unwrap(), order
    else:
        return a.unwrap(), b.unwrap(order), order
