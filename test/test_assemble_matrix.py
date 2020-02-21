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
from petsc4py import PETSc

mesh_list = [dolfinx.UnitIntervalMesh(MPI.COMM_WORLD, 100),
             dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 64, 64),
             dolfinx.UnitCubeMesh(MPI.COMM_WORLD, 10, 10, 10)]


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="This test should only be run in serial.")
@pytest.mark.parametrize("mesh", mesh_list)
def test_assemble_matrix(mesh):
    ''' Test matrix assembly before Dirichlet boundary conditions application.'''
    V = dolfinx.FunctionSpace(mesh, ("Lagrange", 2))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    k = 10 * numpy.pi

    a = ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx + k**2 * ufl.inner(u, v)*ufl.dx \
        + 1j * ufl.inner(u, v) * ufl.ds

    A = odd.create_matrix(a, "communication-less")
    odd.assemble_matrix(a, A)
    A.assemble()

    B = odd.create_matrix(a, "standard")
    odd.assemble_matrix(a, B)
    B.assemble()

    C = dolfinx.fem.assemble_matrix(a)
    C.assemble()

    assert (A-B).norm() < 1e-12
    assert (C-A).norm() < 1e-12


def test_assemble_1d_bc():
    """Solve -u’’ = sin(x), u(0)=0, u(L)=0."""

    comm = MPI.COMM_WORLD
    lim = [0.0, numpy.pi]
    mesh = dolfinx.IntervalMesh(comm, 100, lim)
    mesh.geometry.coord_mapping = dolfinx.fem.create_coordinate_map(mesh)
    mesh.create_connectivity_all()
    tdim = mesh.topology.dim

    [x] = ufl.SpatialCoordinate(mesh.ufl_domain())
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
    values = numpy.zeros(dofs.size)

    # Create matrix
    A = odd.create_matrix(a)
    odd.assemble_matrix(a, A)
    A.assemble()
    odd.apply_bc(A, dofs)

    b = odd.assemble_vector(L)
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

    diff = u - u_e
    e = dolfinx.fem.assemble_scalar(ufl.inner(diff, diff)*ufl.dx)
    print(abs(e))
    assert e == pytest.approx(0)
