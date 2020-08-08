import odd
from odd.sparse.iterative import cg
from scipy.sparse.linalg import cg as spcg
from mpi4py import MPI
import numpy

print("")

comm = MPI.COMM_WORLD
mat_csr = odd.sparse.get_csr_matrix("FEMLAB/poisson2D", verbose=False)
A = odd.sparse.distribute_csr_matrix(mat_csr)

b = A.get_vector()
b.fill(comm.rank)

index_map = b._map
dtype = b._array.dtype
send_data = b._array[index_map.reverse_indices]
recv_data = numpy.zeros(index_map.ghosts.size, dtype=dtype)

mpi_type = MPI._typedict[dtype.char]
displ = numpy.cumsum(index_map.reverse_count())
displ = numpy.insert(displ, 0, 0)[:-1]
index_map.reverse_comm.Neighbor_alltoallv([send_data, (index_map.reverse_count(), displ), mpi_type],
                        [recv_data, index_map.forward_count(), mpi_type])


# x, info = cg(A, b)
# xnorm = numpy.linalg.norm(x)

# # if comm.rank == 0:
# #     x1, info = spcg(mat_csr, numpy.ones(mat_csr.shape[0]))
# #     x1_norm = numpy.linalg.norm(x1)
# #     assert numpy.isclose(xnorm, x1_norm)
