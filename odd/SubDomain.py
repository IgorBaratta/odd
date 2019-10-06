import numpy
from mpi4py import MPI
from dolfin import FunctionSpace, Function, cpp, fem
from dolfin.function import functionspace
from .DofMap import DofMap


class SubDomainData():
    def __init__(self,
                 mesh: cpp.mesh.Mesh,
                 V: functionspace.FunctionSpace,
                 global_comm: MPI.Intracomm):

        # Store dolfin FunctionSpace object
        self._V = V

        # Store MPI communicator
        self.comm = global_comm

        # Create domain decomposition DofMap
        self.dofmap = DofMap(V)

        # Store copy of local mesh
        global_index = mesh.topology.global_indices(mesh.geometry.dim)
        sorted_index = numpy.argsort(global_index).tolist()
        self.mesh = cpp.mesh.Mesh(MPI.COMM_SELF, mesh.cell_type,
                                  mesh.geometry.points[:, :mesh.geometry.dim],
                                  mesh.cells(), sorted_index,
                                  cpp.mesh.GhostMode.none)

        self.mesh.geometry.coord_mapping = fem.create_coordinate_map(self.mesh)
        cpp.mesh.Ordering.order_simplex(self.mesh)

        self._Vi = FunctionSpace(self.mesh, V.ufl_element())

    def restricted_function_space(self):
        return self._Vi

    def global_vec(self):
        u = Function(self._V)
        return u.vector.copy()

    def local2local(self):
        # TODO : Only necessary because local copy of the mesh gets reordered
        # when overlap is added
        dof_oder = numpy.lexsort(self._V.tabulate_dof_coordinates().T)
        dof_oder_i = numpy.lexsort(self._Vi.tabulate_dof_coordinates().T)

        ord_arg = dof_oder.argsort()
        ord_arg_i = dof_oder_i.argsort()

        map_forward = dof_oder_i[ord_arg]
        map_backward = dof_oder[ord_arg_i]
        return map_forward, map_backward

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
