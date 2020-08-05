from mpi4py import MPI
import odd.utils
import odd.sparse
from odd.communication import IndexMap
import numpy

from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg.isolve import cg

comm = MPI.COMM_WORLD
if comm.rank == 0:
    A = odd.sparse.get_csr_matrix("FEMLAB/poisson2D", False)
    b0 = numpy.ones(A.shape[0])
    x0, info = cg(A, b0)
else:
    A = None


# Distribute serial csr matrix by rows, each MPI receive a set of rows
Mat = odd.sparse.distribute_csr_matrix(A, comm)
b = Mat.get_vector()
b.fill(1)

x = Mat @ b

x = Mat.get_vector()
x.fill(0)

from odd.sparse.iterative import cg as cg2

# x = cg(A, b0, x0)
x, info = cg2(Mat, b, x)

print(numpy.max(x.array - x0))
