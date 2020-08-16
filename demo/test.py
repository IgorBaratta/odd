import odd
from odd.sparse.linalg import cg
from scipy.sparse.linalg import spsolve
from mpi4py import MPI
import numpy
import pyamg

from matplotlib import pyplot as plt

comm = MPI.COMM_WORLD

if comm.rank == 0:
    mat = pyamg.gallery.poisson((10000, 10000), format="csr")
else:
    mat = None
A = odd.sparse.distribute_csr_matrix(mat, comm)

b = A.get_vector()
b.fill(1)
x = b.copy()

res = []
t = MPI.Wtime()
for i in range(10):
    x = A @ x
t = MPI.Wtime() - t

if comm.rank == 0:
    print(t)
