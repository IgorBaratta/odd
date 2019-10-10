from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import IntType
import numpy

from dolfin import (DirichletBC, Function, FunctionSpace, TestFunction,
                    TrialFunction, UnitSquareMesh, fem, interpolate)
from dolfin.cpp.mesh import GhostMode
from ufl import SpatialCoordinate, dot, dx, grad, pi, sin

from odd import AdditiveSchwarz, SubDomainData, PETScVectorScatter, RMAVectorScatter


def boundary(x):
    TOL = numpy.finfo(float).eps
    return numpy.logical_or.reduce([x[:, 1] < TOL, x[:, 1] > 1.0 - TOL,
                                    x[:, 0] < TOL, x[:, 0] > 1.0 - TOL])


def solution(values, x):
    values[:, 0] = numpy.sin(numpy.pi*x[:, 0])*numpy.sin(numpy.pi*x[:, 1])


n, p = 10, 1
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

ASM = AdditiveSchwarz(subdomain, A)
ASM.setUp()


local_vec = ASM.vec1.copy()
local_vec.array = comm.rank + 1
global_vec = ASM.vec_global.copy()
global_vec.array = comm.rank + 1

# print(global_vec.array)

dofmap = ASM.dofmap

scatter1 = PETScVectorScatter(comm, dofmap)
scatter2 = RMAVectorScatter(comm, dofmap)

time = MPI.Wtime()
scatter2.reverse(local_vec, global_vec)
time = MPI.Wtime() - time
print(comm.rank, time)
#
# print(local_vec.array)
# print(global_vec.array)
#
#
# disp_unit = global_vec.array.itemsize
# dtype = global_vec.array.dtype
# window = MPI.Win.Create(global_vec.array, disp_unit=disp_unit, comm=comm)
# for i, neighbour in enumerate(dofmap.ghost_owners):
#     offset = dofmap.all_ranges[neighbour]
#     local_index = dofmap.shared_indices[i] - offset
#     buffer = numpy.array(1, dtype=dtype)
#     window.Lock(neighbour, lock_type=MPI.LOCK_SHARED)
#     window.Get([buffer, MPI.DOUBLE], target_rank=neighbour, target=local_index)
#     window.Unlock(neighbour)
#     local_vec.array[dofmap.size_owned + i] = buffer
# window.Free()
#
# print(local_vec.array)
