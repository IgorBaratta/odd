from mpi4py import MPI
import numpy
from dolfin import (FunctionSpace, TestFunction, TrialFunction, UnitSquareMesh, fem)
from dolfin.common import Timer, list_timings, TimingType
from dolfin.cpp.mesh import GhostMode
from ufl import SpatialCoordinate, inner, dx, grad
from odd import AdditiveSchwarz, SubDomainData, ScatterType


n, p = 12, 3
comm = MPI.COMM_WORLD

ghost_mode = GhostMode.shared_vertex if (comm.size > 1) else GhostMode.none
mesh = UnitSquareMesh(comm, 2**n, 2**n, ghost_mode=ghost_mode)

V = FunctionSpace(mesh, ("Lagrange", p))
subdomain = SubDomainData(mesh, V, comm)

u = TrialFunction(V)
v = TestFunction(V)

x, y = SpatialCoordinate(mesh)
f = 10

a = 5*inner(grad(u), grad(v))*dx
L = inner(f, v)*dx

b = fem.assemble_vector(L)
b.assemble()
A = fem.assemble_matrix(a)
A.assemble()

ASM = AdditiveSchwarz(subdomain, A)
ASM.scatter_type = ScatterType.RMA
ASM.setUp()


x = b.duplicate()
y = x.duplicate()

t1 = Timer("xxxxx - ASM")
ASM.mult(None, b, x)
t1.stop()

t2 = Timer("xxxxx- PETSc")
A.mult(b, y)
t2.stop()

# print(y.array)

print(numpy.allclose(x, y))

list_timings([TimingType.wall])
