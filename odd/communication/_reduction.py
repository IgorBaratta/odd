# Copyright (C) 2020 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI
from numbers import Integral
import numpy

MPI_OP = {"amax": MPI.MAX, "amin": MPI.MIN, "sum": MPI.SUM}


def parallel_reduce(dist_array, func=numpy.sum, **kwargs):
    comm: MPI.Intracomm = dist_array.mpi_comm
    local = func(dist_array._array, **kwargs)

    axis = kwargs.get("axis", None)
    if axis is None:
        sendbuf = numpy.asarray(local)
        recvbuf = numpy.zeros(sendbuf.size, dtype=sendbuf.dtype)
        comm.Allreduce(sendbuf, recvbuf, MPI_OP[func.__name__])
        return recvbuf[0]

    elif isinstance(axis, Integral):
        if axis == 0:
            sendbuf = numpy.asarray(local)
            recvbuf = numpy.zeros(sendbuf.size, dtype=sendbuf.dtype)
            comm.Allreduce(sendbuf, recvbuf, MPI_OP[func.__name__])
            return recvbuf
        elif axis == 1:
            return local
    else:
        return NotImplemented
