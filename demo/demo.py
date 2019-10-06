from mpi4py import MPI
from petsc4py import PETSc
import numpy

from dolfin import (DirichletBC, Function, FunctionSpace, TestFunction,
                    TrialFunction, UnitSquareMesh, fem, interpolate)
from dolfin.cpp.mesh import GhostMode
from ufl import SpatialCoordinate, dot, dx, grad, pi, sin

from odd import AdditiveSchwarz, SubDomainData


def boundary(x):
    TOL = numpy.finfo(float).eps
    return numpy.logical_or.reduce([x[:, 1] < TOL, x[:, 1] > 1.0 - TOL,
                                    x[:, 0] < TOL, x[:, 0] > 1.0 - TOL])


def solution(values, x):
    values[:, 0] = numpy.sin(numpy.pi*x[:, 0])*numpy.sin(numpy.pi*x[:, 1])


n, p = 2, 1
comm = MPI.COMM_WORLD

ghost_mode = GhostMode.shared_vertex if (comm.size > 1) else GhostMode.none
mesh = UnitSquareMesh(comm, 2**n, 2**n, ghost_mode=ghost_mode, diagonal="left")

V = FunctionSpace(mesh, ("Lagrange", p))
subdomain = SubDomainData(mesh, V, comm)
# ================================================== #


def assemble(mesh, V):
    u = TrialFunction(V)
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)
    f = 2*pi**2*sin(pi*x)*sin(pi*y)

    a = dot(grad(u), grad(v))*dx
    L = f*v*dx

    # Define boundary condition
    u0 = Function(V)
    u0.vector.set(0.0)
    bcs = [DirichletBC(V, u0, boundary)]

    b = fem.assemble_vector(L)
    fem.apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    A = fem.assemble_matrix(a, bcs)
    A.assemble()
    return A, b
# ===============================================


A, b = assemble(mesh, V)
OSM = AdditiveSchwarz(subdomain, A)
OSM.setUp()
Aa = OSM.global_matrix()
x = A.getVecLeft()

solver = PETSc.KSP().create(comm)
solver.setOperators(Aa)
solver.setType('gmres')
solver.setUp()
solver.pc.setType('python')
solver.pc.setPythonContext(OSM)
solver.setFromOptions()


solver.solve(b, x)

u_exact = interpolate(solution, V)
# print(numpy.linalg.norm(u_exact.vector.array - x.array))


size = x.array.size
itemsize = x.array.itemsize
nbytes = size * itemsize

win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=comm)
buf, itemsize = win.Shared_query(0)



print(nbytes)
# on rank 0, create the shared block
# on rank 1 get a handle to it (known as a window in MPI speak)
# win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=comm)
