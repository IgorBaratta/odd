from mpi4py import MPI
import numpy

from dolfin import (DirichletBC, Function, FunctionSpace, TestFunction,
                    TrialFunction, UnitSquareMesh, fem, cpp, interpolate)
from dolfin.cpp.mesh import GhostMode

from SubDomain import SubDomainData


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
cpp.mesh.Ordering.order_simplex(mesh)

V = FunctionSpace(mesh, ("Lagrange", p))
subdomain = SubDomainData(mesh, V, comm)
Vi = subdomain.restricted_function_space()
Vi2V, V2Vi = subdomain.local2local()
lmesh = subdomain.mesh

# if comm.rank == 0:
#     Vi2V = numpy.zeros(Vi.dim())
#     V2Vi = numpy.zeros(Vi.dim())
#     for i in range(mesh.num_cells()):
#         print(V.dofmap.cell_dofs(i) - Vi.dofmap.cell_dofs(i))


u = Function(V)
ui = Function(Vi)
ui.vector.array = numpy.arange(Vi.dim())

def expression(values, x):
    values[:, 0] = ui(x)

#
# u_exact = interpolate(expression, V)


for i in range(lmesh.num_cells()):
    print(ui.eval([0, 0], i))
