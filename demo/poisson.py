import odd
from odd.sparse.linalg import cg
from scipy.sparse.linalg import cg as spcg
from mpi4py import MPI
import numpy

comm = MPI.COMM_WORLD
mat_csr = odd.sparse.get_csr_matrix("FEMLAB/poisson2D", verbose=False)
A = odd.sparse.distribute_csr_matrix(mat_csr)

b = A.get_vector()
b.fill(1)
