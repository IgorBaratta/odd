from mpi4py import MPI
import numpy
from odd import IndexMap


# Circular domain
comm = MPI.COMM_WORLD
rank = comm.rank
N = 10
n_ghost = 3

neighbor = (rank + 1) % comm.size
ghosts = neighbor*N + numpy.arange(n_ghost)
idx_map = IndexMap(comm, N, ghosts)

assert idx_map.owned_size == N
assert idx_map.local_size == N + n_ghost

assert (rank + 1) % comm.size in idx_map.neighbors
assert (rank - 1) % comm.size in idx_map.neighbors

print(idx_map.shared_indices)
