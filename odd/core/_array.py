# Copyright (C) 2020 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy
from mpi4py import MPI

from collections.abc import Iterable
from functools import reduce
from numbers import Integral

from odd.utils import partition1d
from odd.communication import IndexMap
from ._operations import mpi_reduction, dot1d, vdot1d

from odd.communication.mpi3_scatter import NeighborVectorScatter


HANDLED_FUNCTIONS = {}


class DistArray(numpy.lib.mixins.NDArrayOperatorsMixin):
    def __init__(
        self, shape, dtype=float, buffer=None, index_map=None, comm=MPI.COMM_WORLD
    ):
        """
        Distributed numpy ndarray.

        shape (tuple of ints) – Length of axes.

        """

        # Currently only supports one and two dimensions
        if not isinstance(shape, Iterable):
            shape = (shape,)
        elif len(shape) > 2:
            raise NotImplementedError

        if not all(isinstance(sz, Integral) and int(sz) >= 0 for sz in shape):
            raise ValueError(
                "shape must be an non-negative integer or a tuple "
                "of non-negative integers."
            )

        self.dtype = numpy.dtype(dtype)
        self.mpi_comm = comm
        self.shape = shape

        self._map = None
        if isinstance(index_map, IndexMap):
            self._map = index_map
        elif index_map is None:
            self._map = partition1d(comm, shape[0], 0)

        local_shape = list(shape)
        local_shape[0] = self._map.local_size
        self.local_shape = tuple(local_shape)

        self._array = None
        if buffer is not None:
            if isinstance(buffer, numpy.ndarray):
                self._array = buffer.reshape(self.local_shape)
            else:
                raise TypeError
        else:
            self._array = numpy.ndarray(self.local_shape, dtype=dtype)

        assert self._array.shape[0] == self._map.local_size
        assert self.array.shape[0] == self._map.owned_size

        self.scatter = NeighborVectorScatter(self._map)

    @property
    def array(self):
        return self._array[: self._map.owned_size]

    # @property
    def ghost_values(self):
        return self._array[self._map.owned_size :]

    def shared_values(self):
        return self._array[self._map.forward_indices]

    def update(self):
        """
        Update ghost data.
        Send arrays values from owned indices to, processes
        who has the same index in the ghost region.
        """
        mpi_comm = self._map.forward_comm
        send_data = self.shared_values()
        recv_data = self.ghost_values()

        mpi_comm.Neighbor_alltoallv(
            [send_data, (self._map.forward_count(), None)],
            [recv_data, (self._map.reverse_count(), None)],
        )

    def accumulate(self, op=numpy.add, weights=None):
        """
        Accumulate contributions from ghosts/overlap region
        """
        mpi_comm = self._map.reverse_comm

        if weights is None:
            send_data = self.ghost_values()
        else:
            send_data = weights * self.ghost_values()

        recv_size = numpy.sum(self._map.forward_count())
        recv_data = numpy.zeros(recv_size, dtype=self.dtype)

        mpi_comm.Neighbor_alltoallv(
            [send_data, (self._map.reverse_count(), None)],
            [recv_data, (self._map.forward_count(), None)]
        )

        op.at(self._array, self._map.forward_indices, recv_data)

    def duplicate(self):
        return self.__class__(
            shape=self.shape[0], dtype=self.dtype, index_map=self._map
        )

    def astype(self, dtype):
        return self.__class__(
            shape=self.shape[0],
            dtype=dtype,
            buffer=self._array.astype(dtype),
            index_map=self._map,
        )

    def copy(self):
        return self.__class__(
            shape=self.shape[0],
            dtype=self.dtype,
            buffer=self._array.copy(),
            index_map=self._map,
        )

    def get_local_view(self):
        return self._array

    def get_local_copy(self):
        return self._array.copy()

    def fill(self, value):
        """
        Fill the distributed array with a scalar value.

        @note: the value may be different from each process
        """
        self._array[:] = value

    def __len__(self):
        return reduce((lambda x, y: x * y), self.shape)

    def __repr__(self):
        return (
            f"odd.{self.__class__.__name__}"
            + f"(shape={self.local_shape}, dtype={self.dtype.name},"
            + f" rank={self.mpi_comm.rank})"
        )

    def __setitem__(self, key, value):
        self._array[key] = value

    def __array__(self):
        return self._array[0 : self._map.owned_size]

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Apply unary or binary ufunc to the distributed array,
        currently only supports ufuncs with 1 or 2 inputs and
        only 1 output.

        NumPy will always use it for implementing arithmetic.

        See:
            https://numpy.org/doc/stable/reference/ufuncs.html
        """

        if method == "__call__":
            if ufunc.nin == 1 and ufunc.nout == 1:
                buffer = ufunc(self._array, **kwargs)
                return self.__class__(
                    shape=self.shape,
                    dtype=self.dtype,
                    index_map=self._map,
                    buffer=buffer,
                    comm=self.mpi_comm,
                )

            elif ufunc.nin == 2 and ufunc.nout == 1:
                params = []
                for input in inputs:
                    if isinstance(input, self.__class__):
                        params.append(input._array)
                    else:
                        params.append(input)
                return self.__class__(
                    shape=self.shape,
                    dtype=self.dtype,
                    index_map=self._map,
                    buffer=ufunc(*params, **kwargs),
                    comm=self.mpi_comm,
                )
            else:
                return NotImplemented
        else:
            return NotImplemented

    @property
    def size(self):
        # global size of the distributed array
        return reduce((lambda x, y: x * y), self.shape)

    def allgather(self) -> numpy.ndarray:
        """
        Gathers data from all processes and distributes it to all processes.

        @Note:  Collective Function
        """
        comm = self.mpi_comm
        recvbuf = numpy.zeros(self.shape, dtype=self.dtype)
        sendbuf = self.array.copy()
        comm.Allgatherv(sendbuf, recvbuf)
        return recvbuf

    def __array_function__(self, func, types, args, kwargs):

        if func not in HANDLED_FUNCTIONS:
            return NotImplemented

        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented

        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    @staticmethod
    def get_handled_functions():
        return [func.__name__ for func in HANDLED_FUNCTIONS]


# ========= Handled Operations:


def implements(numpy_func):
    """Register an __array_function__ implementation for odd.ndarray objects."""

    def decorator(func):
        HANDLED_FUNCTIONS[numpy_func] = func
        return func

    return decorator


@implements(numpy.sum)
def sum(array, axis=None, dtype=None):
    return mpi_reduction(array, numpy.sum)


@implements(numpy.min)
def min(array, axis=None, dtype=None):
    return mpi_reduction(array, numpy.min)


@implements(numpy.max)
def max(array, axis=None, dtype=None):
    return mpi_reduction(array, numpy.max)


@implements(numpy.mean)
def mean(array, ord=None, axis=None, dtype=None):
    return mpi_reduction(array, numpy.sum) / array.size


@implements(numpy.linalg.norm)
def norm(array, ord=None, axis=None):
    return mpi_reduction(array, numpy.linalg.norm, ord=ord)


@implements(numpy.size)
def size(array):
    return array.size


@implements(numpy.dot)
def dot(a, b, **kwargs):
    return dot1d(a, b)


@implements(numpy.vdot)
def vdot(a, b, **kwargs):
    return vdot1d(a, b)


@implements(numpy.iscomplexobj)
def iscomplexobj(a, **kwargs):
    return numpy.iscomplexobj(a.array)
