from mpi4py import MPI
from dolfin import Function, FunctionSpace, cpp
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

    def global_vec(self):
        u = Function(self._V)
        return u.vector.copy()

    @property
    def id(self):
        return self.comm.rank
