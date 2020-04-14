from mpi4py import MPI
import numpy

from odd import IndexMap

owned_size = 100

comm = MPI.COMM_WORLD
rank = comm.rank
num_ghosts = owned_size // 10
neighbor = (rank + 1) % comm.size
ghosts = neighbor * owned_size + numpy.arange(num_ghosts)

l2gmap = IndexMap(comm, owned_size, ghosts)

assert l2gmap.owned_size == owned_size
assert l2gmap.local_size == owned_size + num_ghosts

assert (rank + 1) % comm.size in l2gmap.neighbors
assert (rank - 1) % comm.size in l2gmap.neighbors

assert (l2gmap.shared_indices == l2gmap.indices[:l2gmap.num_shared_indices]).all()

print(l2gmap.num_shared_indices)
