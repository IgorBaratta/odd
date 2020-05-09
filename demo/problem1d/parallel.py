import numpy
import odd
from odd.utils import partition1d
from mpi4py import MPI
import jax

comm = MPI.COMM_WORLD
N = 10
ovl = 10

pmap = partition1d(comm, N, ovl)
vec = odd.la.Vector(pmap) + 1j
vec2 = vec + 3

vec3 = numpy.sin(vec)**2 + numpy.cos(vec)**2

print(vec3)
print(vec.vdot(vec) == N)
# print(numpy.sin(vec))