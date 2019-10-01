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


n, p = 3, 2
comm = MPI.COMM_WORLD

ghost_mode = GhostMode.shared_vertex if (comm.size > 1) else GhostMode.none
mesh = UnitSquareMesh(comm, 2**n, 2**n, ghost_mode=ghost_mode, diagonal="left")

V = FunctionSpace(mesh, ("Lagrange", p))
subdomain = SubDomainData(mesh, V, comm)

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


solver = PETSc.KSP().create(comm)
solver.setOperators(A)
solver.setType('gmres')
solver.setUp()
solver.pc.setType('asm')
solver.pc.setASMOverlap(1)
solver.pc.setUp()
# local_ksp = solver.pc.getASMSubKSP()[0]
# local_ksp.setType('preonly')
# local_ksp.pc.setType('lu')
# local_ksp.pc.setFactorSolverType('mumps')
ASM = AdditiveSchwarz(subdomain, A)
solver.pc.setType('python')
solver.pc.setPythonContext(ASM)
solver.setFromOptions()


x = A.getVecLeft()

solver.solve(b, x)

u_exact = interpolate(solution, FunctionSpace(mesh, ("Lagrange", p)))

print(A.getSizes())
print(ASM.PETScSizes())

print(numpy.linalg.norm(u_exact.vector.array - x.array))

print(solver.its)
