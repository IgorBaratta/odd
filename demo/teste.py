from dolfin import (RectangleMesh, FunctionSpace, TrialFunction, TestFunction, MeshFunction,
                    Mesh, cpp)
from dolfin.cpp.mesh import GhostMode, Ordering, CellType
from ufl import dx, grad, inner
import numpy as np
from dolfin.fem import apply_lifting, assemble_matrix, assemble_vector, set_bc
from petsc4py import PETSc
from mpi4py import MPI
from SubDomain import SubDomainData
from AdditiveSchwarz import AdditiveSchwarz
from dolfin.io import XDMFFile
import numpy
from petsc4py import PETSc

from VectorScatter import PETScVectorScatter

# Create mesh read mesh
comm = MPI.COMM_WORLD
Nx, Ny, = 10, 10
p1 = np.array([0., 0., 0.])
p2 = np.array([1., 1., 0.])
cell_type = CellType.triangle
ghost_mode = GhostMode.none
diag = 'left'
mesh = RectangleMesh(comm, [p1, p2], [Nx, Ny], cell_type, ghost_mode, diag)

sorted_index = numpy.argsort(mesh.geometry.global_indices()).tolist()

# =====================================#
V = FunctionSpace(mesh, ("Lagrange", 1))
subdomain = SubDomainData(mesh, V, comm)
Vi = subdomain.restricted_function_space()
interface_facets = subdomain.interface_facets(mesh, True)
ff = MeshFunction("size_t", subdomain.mesh, subdomain.mesh.topology.dim - 1, 0)

if subdomain.id == 0:
    ff.values[interface_facets] = subdomain.id + 1
    enconding = XDMFFile.Encoding.HDF5

    with XDMFFile(subdomain.mesh.mpi_comm(), "interface.xdmf", encoding=enconding) as xdmf:
        xdmf.write(ff)

# ===================================#
u, v = TrialFunction(V), TestFunction(V)

a = inner(grad(u), grad(v)) * dx
L = inner(1.0, v) * dx
A = assemble_matrix(a)
A.assemble()
x, y = A.getVecs()


PC = AdditiveSchwarz(subdomain, A)
PC.setUp()

x.array = comm.rank
# print(PC.vec1.array)
# print(x.array)

local_vec = PC.vec1.copy()
global_vec = x.copy()

vector_scatter = PETSc.Scatter().create(PC.vec1, None, x, PC.is_local)
vector_scatter(x, PC.vec1, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.SCATTER_REVERSE)
print(PC.vec1.array)
local_vec = PC.vec1.copy()
global_vec = x.copy()
dofmap = PC.dofmap

scatter = PETScVectorScatter(dofmap, local_vec, global_vec, comm)
scatter.reverse(local_vec, global_vec)

print(local_vec.array)
