import dolfinx
import odd
import numpy
import ufl
from mpi4py import MPI
from scipy.sparse.linalg import spsolve

comm = MPI.COMM_WORLD
lim = [0.0, numpy.pi]
mesh = dolfinx.IntervalMesh(comm, 100, lim)
mesh.geometry.coord_mapping = dolfinx.fem.create_coordinate_map(mesh)
mesh.create_connectivity_all()
tdim = mesh.topology.dim

[x] = ufl.SpatialCoordinate(mesh.ufl_domain())
f = ufl.sin(x)

V = dolfinx.FunctionSpace(mesh, ("P", 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx

# Define boundary condition on x = lim[0] and x = lim[1]
facets = numpy.where(mesh.topology.on_boundary(tdim-1))[0]
dofs = dolfinx.fem.locate_dofs_topological(V, tdim-1, facets)
values = numpy.zeros(dofs.size)

# Assemble matrix and apply Dirichlet BC
A = odd.fem.assemble_matrix(a).real
odd.fem.apply_bc(A, dofs)

# Assemble vector and apply Dirichlet BC
b = odd.fem.assemble_vector(L)
odd.fem.apply_bc(b, dofs, values)

y = spsolve(A, b)

# Get numerical solution with dolfinx
u0 = dolfinx.Function(V)
u0.vector.set(0.0)
bc = dolfinx.DirichletBC(u0, dofs)
u = dolfinx.Function(V)
dolfinx.solve(a == L, u, bcs=bc)

assert numpy.allclose(y, u.vector.array.real)
