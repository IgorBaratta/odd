# Copyright (C) 2020 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy
import pytest
from mpi4py import MPI
from odd import IndexMap


@pytest.mark.parametrize("owned_size", [10, 50, 100])
def test_circular_domain(owned_size):
    comm = MPI.COMM_WORLD
    rank = comm.rank
    num_ghosts = owned_size/10
    neighbor = (rank + 1) % comm.size
    ghosts = neighbor*owned_size + numpy.arange(num_ghosts)

    idx_map = IndexMap(comm, comm, ghosts)

    assert idx_map.owned_size == owned_size
    assert idx_map.local_size == owned_size + num_ghosts

    assert (rank + 1) % comm.size in idx_map.neighbors
    assert (rank - 1) % comm.size in idx_map.neighbors
