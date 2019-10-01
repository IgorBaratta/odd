from mpi4py import MPI
from petsc4py import PETSc
import numpy

from dolfin import (DirichletBC, Function, FunctionSpace, TestFunction,
                    TrialFunction, UnitSquareMesh, fem, interpolate)
from ufl import SpatialCoordinate, dot, dx, grad, pi, sin


def boundary(x):
    TOL = numpy.finfo(float).eps
    return numpy.logical_or.reduce([x[:, 1] < TOL, x[:, 1] > 1.0 - TOL,
                                    x[:, 0] < TOL, x[:, 0] > 1.0 - TOL])


def solution(values, x):
    values[:, 0] = numpy.sin(numpy.pi*x[:, 0])*numpy.sin(numpy.pi*x[:, 1])


n, p = 4, 2

mesh = UnitSquareMesh(MPI.COMM_WORLD, 2**n, 2**n, diagonal="left")

V = FunctionSpace(mesh, ("Lagrange", p))
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


solver = PETSc.KSP().create(MPI.COMM_WORLD)
solver.setOperators(A)
solver.setType('preonly')
pc = solver.getPC()
pc.setType('lu')
pc.setFactorSolverType('mumps')
solver.setFromOptions()

x = A.getVecLeft()

solver.solve(b, x)


u_exact = interpolate(solution, FunctionSpace(mesh, ("Lagrange", p)))

numpy.linalg.norm(u_exact.vector.array - x.array)
