# Copyright (C) 2020 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import ufl
import dolfinx
import odd
import pytest
from mpi4py import MPI
from petsc4py import PETSc

mesh_list = [dolfinx.UnitIntervalMesh(MPI.COMM_WORLD, 100),
             dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 64, 64),
             dolfinx.UnitCubeMesh(MPI.COMM_WORLD, 10, 10, 10)]


@pytest.mark.parametrize("mesh", mesh_list)
def test_assemble_matrix(mesh):
    ''' Test matrix assembly before Dirichlet boundary conditions application.'''

    V = dolfinx.FunctionSpace(mesh, ("Lagrange", 2))
    dt = odd.SubDomainData(V)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx

    Ai = odd.create_matrix(a)
    odd.assemble_matrix(a, Ai)
    Ai.assemble()

    N = dt.dofmap.size_owned
    Di = dt.partition_of_unity("owned")
    Actx = odd.MatrixContext(Ai, Di)
    A = PETSc.Mat().create()
    A.setType(A.Type.PYTHON)
    A.setSizes(((N, PETSc.DETERMINE), (N, PETSc.DETERMINE)))
    A.setPythonContext(Actx)
    A.setUp()

    B = dolfinx.fem.assemble_matrix(a)
    B.assemble()

    u = dolfinx.Function(V)
    x = u.vector
    x.setRandom()
    y = x.duplicate()
    z = x.duplicate()
    A.mult(x, y)
    B.mult(x, z)

    assert (z-y).norm() == pytest.approx(0)
