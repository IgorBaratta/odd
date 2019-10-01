from mpi4py import MPI
from DofMap import DofMap
from dolfin import FunctionSpace, cpp
from dolfin.function import functionspace
import numpy


class SubDomainData():
    def __init__(self,
                 mesh: cpp.mesh.Mesh,
                 V: functionspace.FunctionSpace,
                 global_comm: MPI.Intracomm):

        self._V = V
        self.dofmap = DofMap(V)
        self.comm = global_comm

        global_index = mesh.topology.global_indices(mesh.geometry.dim)
        sorted_index = numpy.argsort(global_index).tolist()
        self.mesh = cpp.mesh.Mesh(MPI.COMM_SELF, mesh.cell_type,
                                  mesh.geometry.points[:, :mesh.geometry.dim],
                                  mesh.cells(), sorted_index,
                                  cpp.mesh.GhostMode.none)

        self._Vi = FunctionSpace(self.mesh, V.ufl_element())

    def restricted_function_space(self):
        return self._Vi

    def interface_facets(self, mesh, reorder=False):
        gdim = self.mesh.geometry.dim

        boundaries_excl = numpy.array(mesh.topology.on_boundary(gdim - 1))
        boundaries_incl = numpy.array(self.mesh.topology.on_boundary(gdim - 1))

        if reorder:
            midp1 = cpp.mesh.midpoints(mesh, gdim - 1,
                                       range(mesh.num_entities(gdim - 1)))
            midp2 = cpp.mesh.midpoints(self.mesh, gdim - 1,
                                       range(self.mesh.num_entities(gdim - 1)))
            order1 = numpy.lexsort(midp1.T)
            order2 = numpy.lexsort(midp2.T)
            ord_arg = order2.argsort()
            map = order1[ord_arg]
            return numpy.logical_xor(boundaries_excl[map], boundaries_incl)
        else:
            return numpy.logical_xor(boundaries_excl, boundaries_incl)

    @property
    def id(self):
        return self.comm.rank
