import dolfinx
import numpy
import ufl
from mpi4py import MPI
from petsc4py import PETSc

import odd

"""Solve -u’’ = sin(x), u(0)=0, u(L)=0."""

comm = MPI.COMM_WORLD
lim = [0.0, numpy.pi]
mesh = dolfinx.IntervalMesh(comm, 100, lim)
mesh.geometry.coord_mapping = dolfinx.fem.create_coordinate_map(mesh)
mesh.create_connectivity_all()
tdim = mesh.topology.dim

[x] = ufl.SpatialCoordinate(mesh)
f = ufl.sin(x)

V = dolfinx.FunctionSpace(mesh, ("P", 2))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx

# Define boundary condition on x = lim[0] or x = lim[1]
u0 = dolfinx.Function(V)
u0.vector.set(0.0)
facets = numpy.where(mesh.topology.on_boundary(tdim-1))[0]
dofs = dolfinx.fem.locate_dofs_topological(V, tdim-1, facets)

# Create matrix
A = odd.create_matrix(a)
odd.assemble_matrix(a, A)
A.assemble()
odd.apply_bc(A, dofs)

b = dolfinx.fem.assemble_vector(L)
b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
values = numpy.zeros(dofs.size)
odd.apply_bc(b, dofs, values)


ksp = PETSc.KSP().create(comm)
ksp.setOperators(A)
ksp.setType('preonly')
ksp.pc.setType('lu')
ksp.setFromOptions()

u = dolfinx.Function(V)
x = u.vector
ksp.solve(b, x)


u_e = dolfinx.Function(V)
u_e.interpolate(lambda x: numpy.sin(x[0]))
