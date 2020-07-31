from mpi4py import MPI
import numpy

from ._array import DistArray


def empty(shape, map=None, dtype=float, order="C"):
    return DistArray(shape, dtype, index_map=map, comm=MPI.COMM_WORLD)


def zeros(shape, map=None, dtype=float, order="C"):
    array = DistArray(shape, dtype, index_map=map, comm=MPI.COMM_WORLD)
    array.fill(0)
    return array


def ones(shape, map=None, dtype=float, order="C"):
    array = DistArray(shape, dtype, index_map=map, comm=MPI.COMM_WORLD)
    array.fill(0)
    return array


def full(shape, fill_value, map=None, dtype=float, order="C"):
    array = DistArray(shape, dtype, index_map=map, comm=MPI.COMM_WORLD)
    array.fill(fill_value)
    return array
