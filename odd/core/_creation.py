# Copyright (C) 2020 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI

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
