# Copyright (C) 2020 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI
from dolfinx import FunctionSpace, cpp
from .dofmap import DofMap
import numpy
import numba


class SubDomainData():
    def __init__(self,
                 mesh: cpp.mesh.Mesh,
                 V: FunctionSpace,
                 global_comm: MPI.Intracomm):

        # Store dolfinx FunctionSpace object
        self._V = V

        # Store MPI communicator
        self.comm = global_comm

        # Create domain decomposition DofMap
        self.dofmap = DofMap(V)

    @property
    def id(self):
        return self.comm.rank


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
    con_facet_cell = connectivity(tdim - 1, tdim).connections()
    pos_facet_cell = connectivity(tdim - 1, tdim).pos()
    on_boundary = numpy.array(mesh.topology.on_boundary(tdim - 1))
    on_interface = _on_interface(con_facet_cell, pos_facet_cell, on_boundary)
    return on_interface
