import numpy

import odd
from odd.utils import partition1d
from mpi4py import MPI


# Problem data
comm = MPI.COMM_WORLD
global_size = 41
overlap = 2
dx = 1./(global_size - 1)

scatter = odd.NeighborVectorScatter(partition1d(comm, global_size, overlap))

