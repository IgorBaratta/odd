from dolfinx import FunctionSpace, UnitSquareMesh
from dolfinx.cpp.mesh import CellType, GhostMode
import dolfinx
import odd
import ufl
import numpy
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.common import Timer, list_timings, TimingType


comm = MPI.COMM_WORLD
local_comm = MPI.COMM_SELF

ghost_mode = GhostMode.shared_vertex if comm.size > 1 else GhostMode.none
mesh = UnitSquareMesh(comm, 10, 10, CellType.triangle, ghost_mode)
V = FunctionSpace(mesh, ("Lagrange", 3))

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


def _create_cpp_form(form):
    """Recursively look for ufl.Forms and convert to dolfinx.fem.Form, otherwise
    return form argument
    """
    if isinstance(form, dolfinx.Form):
        return form._cpp_object
    elif isinstance(form, ufl.Form):
        return dolfinx.Form(form)._cpp_object
    elif isinstance(form, (tuple, list)):
        return list(map(lambda sub_form: _create_cpp_form(sub_form), form))
    return form


_a = _create_cpp_form(a)
dim = mesh.topology.dim
num_cells = mesh.topology.size(dim)
active_cells = numpy.arange(num_cells)
int_cell = dolfinx.cpp.fem.FormIntegrals.Type.cell
A = dolfinx.cpp.fem.create_matrix(_a)
A.assemble()
indices = V.dofmap.index_map.indices(False).astype(PETSc.IntType)
is_local = PETSc.IS().createGeneral(indices)
Ai = A.createSubMatrices(is_local)[0]
Ai.setUp()

ddm_dofmap = odd.DofMap(V)

N = ddm_dofmap.size_local
self_comm = MPI.COMM_SELF

Ai = PETSc.Mat().create(self_comm)
Ai.setType('seqaij')
Ai.setSizes((N, N))
Ai.setPreallocationNNZ(10)
# Ai.setVal
# dolfinx.cpp.fem.assemble_matrix(Ai, _a, [])
# print(A.type)
