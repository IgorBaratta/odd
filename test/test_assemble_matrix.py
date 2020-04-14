# Copyright (C) 2020 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import ufl
import dolfinx
import numpy
import odd
import pytest
from mpi4py import MPI
from scipy.sparse.linalg import spsolve


mesh_list = [dolfinx.UnitIntervalMesh(MPI.COMM_WORLD, 100),
             dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 10, 10),
             dolfinx.UnitCubeMesh(MPI.COMM_WORLD, 5, 5, 4)]


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="This test should only be run in serial.")
@pytest.mark.parametrize("mesh", mesh_list)
@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_assemble_matrix(mesh, degree):
    """ Test matrix assembly of Helmholtz equation without Dirichlet boundary conditions."""

    mesh.create_connectivity_all()
    V = dolfinx.FunctionSpace(mesh, ("Lagrange", degree))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    k = dolfinx.Constant(mesh, 10 * numpy.pi)

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + k**2 * ufl.inner(u, v) * ufl.dx \
        + 1j * ufl.inner(u, v) * ufl.ds

    A = odd.fem.assemble_matrix(a)

    B = dolfinx.fem.assemble_matrix(a)
    B.assemble()

    ptr, ind, data = B.getValuesCSR()

    assert numpy.all(A.indptr == ptr)
    assert numpy.all(A.indices == ind)
    assert numpy.allclose(A.data, data)


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="This test should only be run in serial.")
@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_assemble_1d_bc(degree):
    """Solve -u’’ = sin(x), u(0)=0, u(L)=0."""

    comm = MPI.COMM_WORLD
    lim = [0.0, numpy.pi]
    mesh = dolfinx.IntervalMesh(comm, 100, lim)
    mesh.create_connectivity_all()
    tdim = mesh.topology.dim

    [x] = ufl.SpatialCoordinate(mesh.ufl_domain())
    f = ufl.sin(x)

    V = dolfinx.FunctionSpace(mesh, ("P", degree))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    # Define boundary condition on x = lim[0] and x = lim[1]
    u0 = dolfinx.Function(V)
    u0.vector.set(0.0)
    facets = numpy.where(mesh.topology.on_boundary(tdim - 1))[0]
    dofs = dolfinx.fem.locate_dofs_topological(V, tdim - 1, facets)
    values = numpy.zeros(dofs.size)

    # Assemble matrix and apply Dirichlet BC
    A = odd.fem.assemble_matrix(a)
    odd.fem.apply_bc(A, dofs)

    # Assemble vector and apply Dirichlet BC
    b = odd.fem.assemble_vector(L)
    odd.fem.apply_bc(b, dofs, values)

    sol = spsolve(A, b)

    # Get numerical solution with dolfinx
    u0 = dolfinx.Function(V)
    u0.vector.set(0.0)
    bc = dolfinx.DirichletBC(u0, dofs)
    u = dolfinx.Function(V)
    dolfinx.solve(a == L, u, bcs=bc)

    assert numpy.allclose(sol.real, u.vector.array.real, atol=1e-05)
