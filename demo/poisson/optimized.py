from dolfinx import FunctionSpace, UnitSquareMesh
from dolfinx.cpp.mesh import CellType, GhostMode
import dolfinx
import odd
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.common import Timer, list_timings, TimingType


comm = MPI.COMM_WORLD
local_comm = MPI.COMM_SELF

ghost_mode = GhostMode.shared_vertex if comm.size > 1 else GhostMode.none
mesh = UnitSquareMesh(comm, 400, 400, CellType.triangle, ghost_mode)
V = FunctionSpace(mesh, ("Lagrange", 4))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

x = ufl.SpatialCoordinate(mesh)
f = ufl.pi**2*ufl.sin(ufl.pi*x[0])*ufl.sin(ufl.pi*x[1])

L = ufl.inner(f, v)*ufl.dx

t1 = Timer("xxxxx - Communication")
b = dolfinx.fem.assemble_vector(L)
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
t1.stop()

t2 = Timer("xxxxx - Communicationless")
b1 = odd.assemble_vector(L)
t2.stop()

print((b - b1).norm())

a = ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx
t3 = Timer("xxxxx - Matrix")
A = dolfinx.fem.assemble_matrix(a)
A.assemble()
t3.stop()

t4 = Timer("xxxxx - Matrix Local")
indices = V.dofmap.index_map.indices(False).astype(PETSc.IntType)
is_local = PETSc.IS().createGeneral(indices)
Ai = A.createSubMatrices(is_local)[0]
t4.stop()

list_timings(comm, [TimingType.wall])
