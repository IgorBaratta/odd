import dolfinx
import numpy
import ufl
from mpi4py import MPI

import odd

mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 10, 10)
degree = 2

V = dolfinx.FunctionSpace(mesh, ("Lagrange", degree))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
k = dolfinx.Constant(mesh, 10 * numpy.pi)

a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + k ** 2 * ufl.inner(u, v) * ufl.dx \
    + 1j * ufl.inner(u, v) * ufl.ds

A = odd.fem.assemble_matrix(a)

B = dolfinx.fem.assemble_matrix(a)
B.assemble()
ptr, ind, data = B.getValuesCSR()

assert numpy.all(A.indptr == ptr)
assert numpy.all(A.indices == ind)
assert numpy.allclose(A.data, data)
