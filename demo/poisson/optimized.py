# Copyright (C) 2020 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfinx.io import XDMFFile
import dolfinx
import numpy
import numba
import time


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


# ghost_mode = dolfinx.cpp.mesh.GhostMode.shared_vertex
ghost_mode = dolfinx.cpp.mesh.GhostMode.none
# mesh = dolfinx.UnitSquareMesh(dolfinx.MPI.comm_world, 20, 20, ghost_mode=ghost_mode)
mesh = dolfinx.UnitCubeMesh(dolfinx.MPI.comm_world, 10, 10, 10)

start = time.time()
on_inter = on_interface(mesh)
end = time.time()
# print("Time (C++, pass 2):", end - start)


tdim = mesh.topology.dim
on_boundary = numpy.array(mesh.topology.on_boundary(tdim - 1))

comm = dolfinx.MPI.comm_world
mf = dolfinx.MeshFunction("size_t", mesh, mesh.topology.dim, 0)
mf.values[:] = comm.rank


with XDMFFile(mesh.mpi_comm(), "mf_1D.xdmf", encoding=XDMFFile.Encoding.HDF5) as file:
        file.write(mf)
