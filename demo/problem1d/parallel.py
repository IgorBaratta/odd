import numpy

import odd
from odd import IndexMap, NeighborVectorScatter, PETScVectorScatter
from odd.utils import partition1d
from mpi4py import MPI

# Problem data
comm = MPI.COMM_WORLD
global_size = 100
overlap = 10
l2gmap = partition1d(comm, global_size, overlap)
scatter = NeighborVectorScatter(l2gmap)
petsc_scatter = PETScVectorScatter(l2gmap)

bi = numpy.ones(l2gmap.local_size, dtype=numpy.complex128) * comm.rank

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
array = bi.copy()
array[l2gmap.owned_size:] = random_ghosts.copy()
scatter.forward(array)

# Update Ghosts with PETSc Vector Scatter
petsc_array = bi.copy()
petsc_array[l2gmap.owned_size:] = random_ghosts.copy()
petsc_scatter.forward(petsc_array)

assert (numpy.all(petsc_array == array))

