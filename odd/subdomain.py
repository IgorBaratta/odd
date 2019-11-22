import numpy
from mpi4py import MPI
from dolfin import Function, FunctionSpace, cpp, fem
from .dofmap import DofMap


class SubDomainData():
    def __init__(self,
                 mesh: cpp.mesh.Mesh,
                 V: FunctionSpace,
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

    @property
    def id(self):
        return self.comm.rank
