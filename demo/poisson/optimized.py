from mpi4py import MPI
from petsc4py import PETSc
import dolfin
import ufl
import numpy
from odd import assemble_vector
from dolfin.cpp.mesh import GhostMode
from dolfin.common import Timer, list_timings, TimingType


def boundary(x):
    TOL = numpy.finfo(float).eps
    return numpy.logical_or.reduce([x[1] < TOL, x[1] > 1.0 - TOL,
                                    x[0] < TOL, x[0] > 1.0 - TOL])


mpi_comm = MPI.COMM_WORLD

ghost_mode = GhostMode.shared_vertex if mpi_comm.size > 1 else GhostMode.none
mesh = dolfin.generation.UnitSquareMesh(mpi_comm, 100, 100, ghost_mode=ghost_mode)
V = dolfin.FunctionSpace(mesh, ("Lagrange", 1))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

x, y = ufl.SpatialCoordinate(mesh)
f = 2*ufl.pi**2*ufl.sin(ufl.pi*x)*ufl.sin(ufl.pi*y)

a = ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx
L = ufl.inner(f, v)*ufl.dx + ufl.inner(1, v)*ufl.dx

t0 = Timer("xxxxx - Assemble Vector - CommunicationLess")
b = assemble_vector(L)
t0.stop()

t1 = Timer("xxxxx - Assemble Vector - Communication")
b1 = dolfin.fem.assemble_vector(L)
b1.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
t1.stop()

print((b - b1).norm())

list_timings(MPI.COMM_WORLD, [TimingType.wall])
