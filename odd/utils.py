# Copyright (C) 2020 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy
from mpi4py import MPI
from .index_map import IndexMap


def partition1d(comm: MPI.Intracomm, global_size: int, overlap: int) -> IndexMap:

    lrange = local_range(comm, global_size)
    ghosts = numpy.array([], dtype=numpy.int64)
    ghost_owner = numpy.array([], dtype=numpy.int32)
    if comm.rank > 0:
        ghosts = numpy.append(ghosts, numpy.arange(lrange[0] - overlap, lrange[0]))
        ghost_owner = numpy.append(ghost_owner, numpy.ones(overlap, dtype=numpy.int32) * (comm.rank - 1))
    if comm.rank < comm.size - 1:
        ghosts = numpy.append(ghosts, numpy.arange(lrange[1], lrange[1] + overlap))
        ghost_owner = numpy.append(ghost_owner, numpy.ones(overlap, dtype=numpy.int32) * (comm.rank + 1))
    owned_size = lrange[1] - lrange[0]

    return IndexMap(comm, owned_size, ghosts, ghost_owner)


def local_range(comm: MPI.Intracomm, global_size: int):
    n = global_size // comm.size
    r = global_size % comm.size
    if comm.rank < r:
        lrange = [comm.rank * (n + 1), comm.rank * (n + 1) + n + 1]
    else:
        lrange = [comm.rank * n + r, comm.rank * n + r + n]
    return lrange