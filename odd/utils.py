# Copyright (C) 2020 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy
from mpi4py import MPI
from .index_map import IndexMap


def partition1d(comm, global_size, overlap):
    lrange = local_range(comm, global_size)
    ghosts = numpy.array([], dtype=numpy.int64)
    if comm.rank > 0:
        ghosts = numpy.append(ghosts, numpy.arange(lrange[0] - overlap, lrange[0]))
    if comm.rank < comm.size - 1:
        ghosts = numpy.append(ghosts, numpy.arange(lrange[1], lrange[1] + overlap))
    osize = owned_size(comm, global_size)
    return IndexMap(comm, osize, ghosts)


def local_range(comm: MPI.Intracomm, global_size: int):
    n = global_size // comm.size
    r = global_size % comm.size
    if comm.rank < r:
        lrange = [comm.rank * (n + 1), comm.rank * (n + 1) + n + 1]
    else:
        lrange = [comm.rank * n + r, comm.rank * n + r + n]
    return lrange


def owned_size(comm: MPI.Intracomm, global_size: int):
    n = global_size // comm.size
    r = global_size % comm.size
    return n + 1 if comm.rank < r else n + r - 1
