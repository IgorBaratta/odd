from dolfinx import FunctionSpace, MPI, UnitSquareMesh
from dolfinx.cpp.mesh import CellType, GhostMode
import dolfinx
import ufl

comm = MPI.comm_world
ghost_mode = GhostMode.shared_vertex if comm.size > 1 else GhostMode.none

mesh = UnitSquareMesh(MPI.comm_world, 4, 4, CellType.triangle, ghost_mode)
V = FunctionSpace(mesh, ("Lagrange", 1))

print(V.dofmap.dof_array)
print(mesh.num_entities(2))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

a = ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx
A = dolfinx.fem.create_matrix(a)

print(A.size)
