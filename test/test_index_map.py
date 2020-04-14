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
@pytest.mark.skipif(MPI.COMM_WORLD.size == 1,
                    reason="This test should only be run in parallel.")
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
    assert (l2gmap.shared_indices == l2gmap.indices[:l2gmap.num_shared_indices]).all()



