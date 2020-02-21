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
