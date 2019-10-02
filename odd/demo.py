from mpi4py import MPI
from petsc4py import PETSc
import numpy

from dolfin import (DirichletBC, Function, FunctionSpace, TestFunction,
                    TrialFunction, UnitSquareMesh, fem, interpolate)
from dolfin.cpp.mesh import GhostMode
from ufl import SpatialCoordinate, dot, dx, grad, pi, sin

from SubDomain import SubDomainData
from AdditiveSchwarz import AdditiveSchwarz


def boundary(x):
    TOL = numpy.finfo(float).eps
    return numpy.logical_or.reduce([x[:, 1] < TOL, x[:, 1] > 1.0 - TOL,
                                    x[:, 0] < TOL, x[:, 0] > 1.0 - TOL])


def solution(values, x):
    values[:, 0] = numpy.sin(numpy.pi*x[:, 0])*numpy.sin(numpy.pi*x[:, 1])


n, p = 3, 1
comm = MPI.COMM_WORLD

ghost_mode = GhostMode.none
mesh = UnitSquareMesh(comm, 2**n, 2**n, ghost_mode=ghost_mode, diagonal="left")

V = FunctionSpace(mesh, ("Lagrange", p))
subdomain = SubDomainData(mesh, V, comm)
Vi = subdomain.restricted_function_space()
lmesh = subdomain.mesh

u = TrialFunction(Vi)
v = TestFunction(Vi)

x, y = SpatialCoordinate(lmesh)
f = 2*pi**2*sin(pi*x)*sin(pi*y)

a = dot(grad(u), grad(v))*dx
L = f*v*dx

# Define boundary condition
u0 = Function(Vi)
u0.vector.set(0.0)
bcs = [DirichletBC(Vi, u0, boundary)]

bi = fem.assemble_vector(L)
fem.apply_lifting(bi, [a], [bcs])
Ai = fem.assemble_matrix(a, bcs)
Ai.assemble()

OSM = AdditiveSchwarz(subdomain, Ai)
A = PETSc.Mat().create()
A.setSizes(*OSM.PETScSizes())
A.setType(A.Type.PYTHON)
A.setPythonContext(OSM)
A.setUp()


solver = PETSc.KSP().create(comm)
solver.setOperators(A)
solver.setType('gmres')
solver.setUp()
solver.pc.setType('python')
solver.pc.setPythonContext(OSM)
solver.setFromOptions()
#
#
# x = A.getVecLeft()
#
# solver.solve(b, x)
#
# u_exact = interpolate(solution, FunctionSpace(mesh, ("Lagrange", p)))
# print(numpy.linalg.norm(u_exact.vector.array - x.array))
