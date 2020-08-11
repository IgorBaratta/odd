import odd
from odd.sparse.linalg import cg
from scipy.sparse.linalg import spsolve
from mpi4py import MPI
import numpy
import pyamg

from matplotlib import pyplot as plt

comm = MPI.COMM_WORLD

if comm.rank == 0:
    mat = pyamg.gallery.poisson((500, 500), format="csr")
else:
    mat = None
A = odd.sparse.distribute_csr_matrix(mat, comm)

b = A.get_vector()
b.fill(1)

res = []
t = MPI.Wtime()
x, info = cg(A, b, residuals=res)
t = MPI.Wtime() - t

# print(numpy.linalg.norm(x))
if comm.rank == 0:
    plt.plot(numpy.log10(res))
    plt.show()
    print(res[-1])
    print(t)
