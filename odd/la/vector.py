# Copyright (C) 2020 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import collections
from numbers import Number

import numpy
from mpi4py import MPI
from odd.index_map import IndexMap
from odd.vector_scatter import NeighborVectorScatter

_HANDLED_TYPES = (numpy.ndarray, collections.Sequence, Number)


class Vector(numpy.lib.mixins.NDArrayOperatorsMixin):
    r"""
    Parallel odd Vector. Distributed memory 1d array.
    Can be viewed as a Distributed extension to one dimensional numpy.ndarray.
    The parallel layout of Vector is handled by an IndexMap.

    (Almost) All numpy elementwise operations works seamlessly on odd.Vector.
    """
    ndim = 1
    __array_priority__ = 11  # higher than numpy.ndarray and numpy.matrix

    def __init__(self, parallel_partitioning: IndexMap, values=None, dtype=None):
        if not isinstance(parallel_partitioning, IndexMap):
            raise TypeError("Invalid parallel partition map")

        self.map = parallel_partitioning
        self.scatter = NeighborVectorScatter(self.map)

        if values is None:
            self._array = numpy.zeros(self.map.local_size)
        elif isinstance(values, _HANDLED_TYPES):
            if len(values) == self.map.local_size:
                self._array = numpy.array(values)
            elif len(values) == self.map.owned_size:
                values = numpy.array(values)
                self._array = numpy.zeros(self.map.local_size, dtype=values.dtype)
                self._array[:self.map.owned_size] = values
            else:
                raise ValueError("Wrong Size")
        else:
            raise TypeError("Not recognized type for array")

        # Fixme:
        if dtype and dtype != self._array.dtype:
            self._array.astype(dtype)
        self.dtype = self._array.dtype

    def __array__(self, dtype=None, **kwargs):
        array = self.local_array
        if dtype and array.dtype != dtype:
            array = array.astype(dtype)
        return array

    @property
    def local_array(self):
        """
        Return a view to the owned part of the array.
        """
        return self._array[:self.map.owned_size]

    def array(self):
        return self._array

    def norm(self) -> float:
        pass

    def copy(self):
        """
        Copy Vector.
        """
        return Vector(self.map, self._array, dtype=self.dtype)

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        v = self.copy()
        memodict[id(self)] = v
        return v

    def __str__(self):
        return str(self.map.comm.rank) + " " + self.local_array.__str__()

    def astype(self, dtype, **kwargs):
        return Vector(self.map, self._array, dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        new_inputs = []
        dtype = kwargs.get("dtype", self.dtype)
        if method == '__call__':
            for x in inputs:
                if isinstance(x, self.__class__):
                    new_inputs.append(x._array)
                elif isinstance(x, _HANDLED_TYPES):
                    new_inputs.append(x)
                else:
                    return NotImplemented
            return self.__class__(self.map, ufunc(*new_inputs, **kwargs), dtype)
        else:
            return NotImplemented

    def dot(self, other):
        """
        Return the inner product of vectors (without complex conjugation).
        The result is reduced to all
        """
        local_dot = numpy.asarray(numpy.dot(self, other))
        recv_buffer = numpy.array(1, dtype=self.dtype)
        self.map.comm.Allreduce(local_dot, recv_buffer, MPI.SUM)
        return recv_buffer

    def vdot(self, other):
        """
        Return the dot product of two vectors.
        Conjugate self
        """
        local_dot = numpy.array(numpy.dot(numpy.conj(self), other))
        recv_buffer = numpy.array(1, dtype=self.dtype)
        self.map.comm.Allreduce(local_dot, recv_buffer, MPI.SUM)
        return recv_buffer

    @property
    def shape(self) -> tuple:
        return self._array.shape

    def __len__(self):
        return len(self._array)

    def __setitem__(self, key, value):
        self._array[key] = value

    def __getitem__(self, key):
        return self._array[key]

    def __setslice__(self, i, j, sequence):
        self._array[i:j] = sequence

    def __iter__(self):
        return iter(self._array)
