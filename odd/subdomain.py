# Copyright (C) 2020 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfinx import FunctionSpace
from .index_map import IndexMap
from petsc4py import PETSc
from mpi4py import MPI
import numpy
import numba


class SubDomainData():
    def __init__(self, V: FunctionSpace):

        # Store dolfinx FunctionSpace object
        self._V = V

        # Store MPI communicator
        self.comm = V.mesh.mpi_comm()

        # Create domain decomposition DofMap
        self.indexmap = IndexMap(V)

    @property
    def id(self):
        return self.comm.rank

    def restritction_matrix(self) -> PETSc.Mat:
        """
        Explicitely construct the local restriction matrix for
        the current subdomain.
        Good for testing.
        """

        # number of non-zeros per row
        nnz = 1

        # Local Size, including overlap
        N = self.dofmap.size_local

        # Global Size
        N_global = self.dofmap.size_global

        # create restriction data in csr format
        A = numpy.ones(N, dtype=PETSc.IntType)
        IA = numpy.arange(N + 1, dtype=PETSc.IntType)
        JA = self.dofmap.indices

        # Create and assembly local Restriction Matrix
        R = PETSc.Mat().create(MPI.COMM_SELF)
        R.setType('aij')
        R.setSizes([N, N_global])
        R.setPreallocationNNZ(nnz)
        R.setValuesCSR(IA, JA, A)
        R.assemblyBegin()
        R.assemblyEnd()

        return R

    def partition_of_unity(self, mode="owned") -> PETSc.Mat:
        """
        Return the assembled partition of unit matrix for the current
        subdomain/process.
        Good for testing.
        """

        # number of non-zeros per row
        nnz = 1
        N = self.dofmap.size_local
        N_owned = self.dofmap.size_owned

        # create restriction data in csr format
        A = numpy.zeros(N, dtype=PETSc.IntType)
        A[0:N_owned] = 1
        IA = numpy.arange(N + 1, dtype=PETSc.IntType)
        JA = numpy.arange(N, dtype=PETSc.IntType)

        # Create and assemble Partition of Unity Matrix
        D = PETSc.Mat().create(MPI.COMM_SELF)
        D.setType('aij')
        D.setSizes([N, N])
        D.setPreallocationNNZ(nnz)
        D.setValuesCSR(IA, JA, A)
        D.assemblyBegin()
        D.assemblyEnd()

        return D


@numba.njit(fastmath=True)
def _on_interface(con_facet_cell, pos_facet_cell, on_boundary):
    on_interface = numpy.zeros_like(on_boundary)
    internal_facets = numpy.where(numpy.logical_not(on_boundary))[0]
    for facet in internal_facets:
        cells = con_facet_cell[pos_facet_cell[facet]: pos_facet_cell[facet + 1]]
        if cells.size < 2:
            on_interface[facet] = True
    return on_interface


def on_interface(mesh):
    tdim = mesh.topology.dim
    connectivity = mesh.topology.connectivity
    con_facet_cell = connectivity(tdim - 1, tdim).array()
    pos_facet_cell = connectivity(tdim - 1, tdim).offsets()
    on_boundary = numpy.array(mesh.topology.on_boundary(tdim - 1))
    on_interface = _on_interface(con_facet_cell, pos_facet_cell, on_boundary)
    return on_interface
