# Copyright (C) 2020 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy
import pytest
from mpi4py import MPI
from odd import IndexMap, NeighborVectorScatter, PETScVectorScatter
from odd.utils import partition1d


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

    assert numpy.all(l2gmap.reverse_indices == numpy.arange(num_ghosts))


@pytest.mark.parametrize("global_size", [50, 100])
@pytest.mark.parametrize("overlap", [2, 5, 10])
@pytest.mark.skipif(MPI.COMM_WORLD.size == 1,
                    reason="This test should only be run in parallel.")
def test_vec_scatter(global_size, overlap):
    # Problem data
    comm = MPI.COMM_WORLD

    l2gmap = partition1d(comm, global_size, overlap)
    scatter = NeighborVectorScatter(l2gmap)
    petsc_scatter = PETScVectorScatter(l2gmap)

    bi = numpy.ones(l2gmap.local_size) * comm.rank

    # Update Ghosts with MPI-3 Neighborhood Communication
    array = bi.copy()
    array[l2gmap.owned_size:] = 0
    scatter.reverse(array)

    # Update Ghosts with PETSc Vector Scatter
    index_map = l2gmap
    petsc_array = bi.copy()
    petsc_array[index_map.owned_size:] = 0
    petsc_scatter.reverse(petsc_array)

    assert (numpy.all(petsc_array == array))

    # Update Ghosts with MPI-3 Neighborhood Communication
    random_ghosts = numpy.random.rand(l2gmap.ghosts.size)
    array = bi.astype(numpy.complex128)
    array[l2gmap.owned_size:] = random_ghosts
    scatter.forward(array)

    # Update Ghosts with PETSc Vector Scatter
    petsc_array = bi.astype(numpy.complex128)
    petsc_array[l2gmap.owned_size:] = random_ghosts
    petsc_scatter.forward(petsc_array)

    assert (numpy.all(petsc_array == array))
