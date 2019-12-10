import numpy
from mpi4py import MPI
from petsc4py import PETSc

from dolfin import (DirichletBC, Function, FunctionSpace, UnitSquareMesh, fem)
from dolfin.common import Timer, list_timings, TimingType
from dolfin.cpp.mesh import GhostMode
from ufl import SpatialCoordinate, TestFunction, TrialFunction, inner, dx, grad, pi, sin


def boundary(x):
    TOL = numpy.finfo(float).eps
    return numpy.logical_or.reduce([x[1] < TOL, x[1] > 1.0 - TOL,
                                    x[0] < TOL, x[0] > 1.0 - TOL])


def solution(x):
    return numpy.sin(numpy.pi*x[0])*numpy.sin(numpy.pi*x[1])


n, p = 8, 2
comm = MPI.COMM_WORLD

ghost_mode = GhostMode.shared_vertex if (comm.size > 1) else GhostMode.none
mesh = UnitSquareMesh(comm, 2**n, 2**n, ghost_mode=ghost_mode)

V = FunctionSpace(mesh, ("Lagrange", p))

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


solver = PETSc.KSP().create(comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.GMRES)
solver.setUp()
solver.pc.setType(PETSc.PC.Type.ASM)
solver.pc.setASMOverlap(1)
solver.pc.setUp()
local_ksp = solver.pc.getASMSubKSP()[0]
local_ksp.setType(PETSc.KSP.Type.PREONLY)
local_ksp.pc.setType(PETSc.PC.Type.LU)
local_ksp.pc.setFactorSolverType('mumps')


x = A.getVecLeft()

t1 = Timer("xxxxx - Solve")
solver.solve(b, x)
t1.stop()

u_exact = Function(V)
u_exact.interpolate(solution)

if comm.rank == 0:
    print("==================================")
    print("Number of Subdomains: ", comm.size)
    print("Error Norm:", numpy.linalg.norm(u_exact.vector.array - x.array))
    print("Number of Iterations:", solver.its)
    print("==================================")

list_timings(MPI.COMM_WORLD, [TimingType.wall])
