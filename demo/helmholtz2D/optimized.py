# Copyright (C) 2020 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import dolfinx
import odd
import ufl
import numpy
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.io import XDMFFile
from dolfinx.common import Timer, list_timings, TimingType

arr = numpy.linspace(-2, 2, 20)

comm = MPI.COMM_WORLD

# Free-Space wavenumber
k0 = 5*numpy.pi


# Generate mesh
shared_vertex = dolfinx.cpp.mesh.GhostMode.shared_vertex
none = dolfinx.cpp.mesh.GhostMode.none
ghost_mode = shared_vertex if (comm.size > 1) else none
mesh = dolfinx.UnitSquareMesh(comm, 32, 32, ghost_mode=none)
n = dolfinx.FacetNormal(mesh)


def plane_wave(x):
    '''Plane Wave Expression'''
    theta = numpy.pi/4
    return numpy.exp(1.0j * k0 * (numpy.cos(theta) * x[0] + numpy.sin(theta) * x[1]))


for j in arr:
    for k in arr:

        # Definition of test and trial function spaces
        deg = 1  # polynomial degree
        V = dolfinx.FunctionSpace(mesh, ("Lagrange", deg))
        subdomain = odd.SubDomainData(mesh, V, comm)

        # Prepare Expression as FE function
        ui = dolfinx.Function(V)
        ui.interpolate(plane_wave)
        g = ufl.dot(ufl.grad(ui), n) + 1j * k0 * ui

        # Define variational problem
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k0**2 * ufl.inner(u, v) * ufl.dx  \
            + 1j * k0 * ufl.inner(u, v) * ufl.ds
        L = ufl.inner(g, v) * ufl.ds

        b = dolfinx.fem.assemble_vector(L)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        A = odd.create_matrix(a)
        odd.assemble_matrix(a, A)

        alpha = (j + k*1j) * k0
        TC = alpha * ufl.inner(u, v) * ufl.ds

        if abs(alpha) > 0.00001:
            odd.apply_transmission_condition(A, TC)
            A.assemble()

            ASM = odd.AdditiveSchwarz(subdomain, A)
            ASM.setUp()

            C = dolfinx.fem.assemble_matrix(a)
            C.assemble()

            ksp = PETSc.KSP().create(comm)
            ksp.setOperators(C)
            ksp.setType(PETSc.KSP.Type.GMRES)
            # ksp.setTolerances(rtol=1e-6)
            ksp.pc.setType(PETSc.PC.Type.PYTHON)
            ksp.pc.setPythonContext(ASM)
            ksp.setFromOptions()

            x = b.duplicate()

            t1 = Timer("xxxxx - Solve")
            ksp.solve(b, x)
            t1.stop()

            u_exact = dolfinx.Function(V)
            u_exact.interpolate(plane_wave)

            u = dolfinx.Function(V)
            x.copy(u.vector)

            # Compute error norm
            diff = u - u_exact
            error = numpy.array(abs(dolfinx.fem.assemble_scalar(ufl.inner(diff, diff)*ufl.dx)))
            error_norm = numpy.zeros(1)
            comm.Reduce(error, error_norm)

            if comm.rank == 0:
                print("==================================")
                print("k0 ===", alpha)
                print("Number of Subdomains: ", comm.size)
                print("Error Norm:", error_norm)
                print("Number of Iterations:", ksp.its)
                print("==================================")

            ksp.destroy()
#
# list_timings(MPI.COMM_WORLD, [TimingType.wall])
#
# file = "file.xdmf"
# encoding = XDMFFile.Encoding.HDF5
# with XDMFFile(mesh.mpi_comm(), file, encoding=encoding) as file:
#         file.write(u)
