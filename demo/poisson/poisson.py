from mpi4py import MPI
from petsc4py import PETSc
import numpy

from dolfin import (DirichletBC, Function, FunctionSpace, TestFunction,
                    TrialFunction, UnitSquareMesh, fem)
from dolfin.cpp.mesh import GhostMode
from dolfin.common import Timer, list_timings, TimingType
from ufl import SpatialCoordinate, inner, dx, grad, pi, sin

from odd import AdditiveSchwarz, SubDomainData


def boundary(x):
    TOL = numpy.finfo(float).eps
    return numpy.logical_or.reduce([x[:, 1] < TOL, x[:, 1] > 1.0 - TOL,
                                    x[:, 0] < TOL, x[:, 0] > 1.0 - TOL])


def solution(x):
    return numpy.sin(numpy.pi*x[:, 0])*numpy.sin(numpy.pi*x[:, 1])


n, p = 8, 2
comm = MPI.COMM_WORLD

ghost_mode = GhostMode.shared_vertex if (comm.size > 1) else GhostMode.none
mesh = UnitSquareMesh(comm, 2**n, 2**n, ghost_mode=ghost_mode)

V = FunctionSpace(mesh, ("Lagrange", p))
subdomain = SubDomainData(mesh, V, comm)

u = TrialFunction(V)
v = TestFunction(V)

x, y = SpatialCoordinate(mesh)
f = 2*pi**2*sin(pi*x)*sin(pi*y)

a = inner(grad(u), grad(v))*dx
L = inner(f, v)*dx

# Define boundary condition
u0 = Function(V)
u0.vector.set(0.0)
bcs = [DirichletBC(V, u0, boundary)]

b = fem.assemble_vector(L)
fem.apply_lifting(b, [a], [bcs])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
A = fem.assemble_matrix(a, bcs)
A.assemble()

ASM = AdditiveSchwarz(subdomain, A)
ASM.setUp()

ksp = PETSc.KSP().create(comm)
ksp.setOperators(A)
ksp.setType('gmres')
ksp.pc.setType('python')
ksp.pc.setPythonContext(ASM)
ksp.setFromOptions()

t1 = Timer("xxxxx - Solve")
x = b.duplicate()
ksp.solve(b, x)
t1.stop()

u_exact = Function(V)
u_exact.interpolate(solution)
if comm.rank == 0:
    print(numpy.linalg.norm(u_exact.vector.array - x.array))
    print(ksp.its)

list_timings([TimingType.wall])
