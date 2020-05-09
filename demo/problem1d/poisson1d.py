import dolfinx
import numpy
import ufl
from mpi4py import MPI
from petsc4py import PETSc
import time

import odd

"""Solve -u’’ = sin(x), u(0)=0, u(L)=0."""

comm = MPI.COMM_WORLD
lim = [0.0, numpy.pi]
ghost_mode = dolfinx.cpp.mesh.GhostMode.shared_facet
mesh = dolfinx.IntervalMesh(comm, 10000, lim, ghost_mode)
mesh.topology.create_connectivity_all()
tdim = mesh.topology.dim

V = dolfinx.FunctionSpace(mesh, ("P", 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
[x] = ufl.SpatialCoordinate(mesh)
f = ufl.sin(x)
L = ufl.inner(f, v) * ufl.dx


# Using dolfinx/PETSc
A = dolfinx.fem.assemble_matrix(a)
A.assemble()
u_e = dolfinx.Function(V)
u_e.interpolate(lambda y: numpy.sin(y[0]))
tic = time.perf_counter()
v = A*u_e.vector
toc = time.perf_counter()
# print(f"PETSc in {toc - tic:0.4f} seconds")
# v.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

# Using odd
size_local = V.dofmap.index_map.size_local
ghosts = V.dofmap.index_map.ghosts
imap = odd.IndexMap(comm, size_local, ghosts)
Ai = odd.fem.assemble_matrix(a)
Op = odd.la.LinearOperator(Ai, imap)

with u_e.vector.localForm() as local_vec:
    vec = odd.la.Vector(imap, local_vec._array, PETSc.ScalarType)

c = Op(vec)
tic = time.perf_counter()
c[:] = numpy.sin(vec._array)
toc = time.perf_counter()
print(f"Odd in {toc - tic:0.4f} seconds")
# print(numpy.sum((c.local_array - v.array)))
#
print(type(c))