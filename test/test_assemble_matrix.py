import ufl
import dolfinx
import numpy
import odd


def test_assemble_matrix():
    mesh = dolfinx.UnitSquareMesh(dolfinx.MPI.comm_world, 100, 100)
    V = dolfinx.FunctionSpace(mesh, ("Lagrange", 1))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    k = 10 * numpy.pi

    a = ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx + k**2 * ufl.inner(u, v)*ufl.dx \
        + 1j * ufl.inner(u, v) * ufl.ds

    A = odd.create_matrix(a, "communication-less")
    odd.assemble_matrix(a, A)
    A.assemble()

    B = odd.create_matrix(a, "standard")
    odd.assemble_matrix(a, A)
    B.assemble()

    C = dolfinx.fem.assemble_matrix(a)
    C.assemble()

    assert (A-B).norm() < 1e-12
    assert (C-A).norm() < 1e-12
