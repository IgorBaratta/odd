# Copyright (C) 2020 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy
import pytest
from mpi4py import MPI
from odd.communication import IndexMap, NeighborVectorScatter
from odd.utils import partition1d

import odd


@pytest.mark.parametrize("owned_size", [10, 50, 100])
@pytest.mark.skipif(
    MPI.COMM_WORLD.size == 1, reason="This test should only be run in parallel."
)
def test_circular_domain(owned_size):
    comm = MPI.COMM_WORLD
    rank = comm.rank
    num_ghosts = owned_size / 10
    neighbor = (rank + 1) % comm.size
    ghosts = neighbor * owned_size + numpy.arange(num_ghosts)

    l2gmap = IndexMap(comm, owned_size, ghosts)

    assert l2gmap.owned_size == owned_size
    assert l2gmap.local_size == owned_size + num_ghosts

    assert (rank + 1) % comm.size in l2gmap.neighbors
    assert (rank - 1) % comm.size in l2gmap.neighbors

    assert numpy.all(numpy.sort(l2gmap.forward_indices) == numpy.arange(num_ghosts))


@pytest.mark.parametrize("global_size", [100, 200])
@pytest.mark.parametrize("overlap", [2, 5, 10])
@pytest.mark.skipif(
    MPI.COMM_WORLD.size == 1, reason="This test should only be run in parallel."
)
def test_vec_scatter(global_size, overlap):
    comm = MPI.COMM_WORLD
    l2gmap = partition1d(comm, global_size, overlap)

    b = odd.DistArray(global_size, index_map=l2gmap)
    b.fill(comm.rank)

    # before update
    assert numpy.all(b.ghost_values() == comm.rank)

    b.update()

    assert numpy.all(b.ghost_values() == b._map.ghost_owners)
