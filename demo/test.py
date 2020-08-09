from mpi4py import MPI

import numpy

from odd.utils import partition1d
from odd.communication import NeighborVectorScatter

global_size = 100
overlap = 10

comm = MPI.COMM_WORLD
l2gmap = partition1d(comm, global_size, overlap)
scatter = NeighborVectorScatter(l2gmap)

bi = numpy.ones(l2gmap.local_size, dtype=numpy.float) * comm.rank

# Update Ghosts with MPI-3 Neighborhood Communication
array = bi.copy()
array[l2gmap.owned_size :] = 0
scatter.reverse(array)
assert numpy.all(array[l2gmap.owned_size :] == l2gmap.ghost_owners)

