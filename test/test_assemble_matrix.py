import ufl
import dolfinx
import numpy
import odd


def test_assemble_matrix():
    mesh = dolfinx.UnitSquareMesh(dolfinx.MPI.comm_self, 100, 100)
    V = dolfinx.FunctionSpace(mesh, ("Lagrange", 1))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    k = 10 * numpy.pi

    a = ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx + k * ufl.inner(u, v)*ufl.dx \
        + 1j * ufl.inner(u, v) * ufl.ds

    A = odd.assemble_matrix(a)
    A.assemble()

    B = dolfinx.fem.assemble_matrix(a)
    B.assemble()

    assert (A-B).norm() < 1e-12
