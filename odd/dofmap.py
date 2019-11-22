from dolfin.function import FunctionSpace
from petsc4py.PETSc import IntType
# from mpi4py import MPI
import numpy


class DofMap:
    """ TODO : class docstring """
    def __init__(self,
                 V: FunctionSpace):
        """ All communication within this class should
        be done at __init__"""
        self._dofmap = V.dofmap
        self._index_map = V.dofmap.index_map
        self.comm = V.mesh.mpi_comm()

        self._all_ranges = self.comm.allgather(self._index_map.size_local)
        self._all_ranges = numpy.cumsum(self._all_ranges)
        self._all_ranges = numpy.insert(self._all_ranges, 0, 0)

    @property
    def id(self):
        return self.comm.rank

    @property
    def indices(self) -> numpy.ndarray:
        """ Return array of global indices for all dofs on this process,
        including shared dofs"""
        return self._index_map.indices(True).astype(IntType)

    @property
    def owned_indices(self) -> numpy.ndarray:
        """ Return array of global indices for the owned dofs on this process,
        not including shared dofs"""
        return self.indices[:self.size_local]

    @property
    def shared_indices(self) -> numpy.ndarray:
        """ Return array of global indices for the shared dofs on this
        process."""
        return self.indices[self.size_owned:]

    @property
    def size_local(self) -> int:
        """ Returns a string to be used as a printable representation
        of a given rectangle."""
        return self._index_map.size_local + self._index_map.num_ghosts

    @property
    def size_owned(self) -> int:
        """ Returns a string to be used as a printable representation
        of a given rectangle."""
        return self._index_map.size_local

    @property
    def size_overlap(self) -> int:
        """ Returns a string to be used as a printable representation
        of a given rectangle."""
        return self._index_map.num_ghosts

    @property
    def size_global(self) -> int:
        """ Returns a string to be used as a printable representation
        of a given rectangle."""
        return self._index_map.size_global

    @property
    def ghost_owners(self):
        """ Owner rank of each ghost entry"""
        return self._index_map.ghost_owners

    @property
    def neighbours(self):
        """ Returns a string to be used as a printable representation
        of a given rectangle."""
        return numpy.unique(self.ghost_owners)

    @property
    def ghosts(self):
        """ Returns a string to be used as a printable representation
        of a given rectangle."""
        return self._index_map.ghosts.astype(IntType)

    @property
    def all_ranges(self):
        """ Returns a string to be used as a printable representation
        of a given rectangle."""
        return self._all_ranges

    def neighbour_ghosts(self, i: int) -> (numpy.ndarray, numpy.ndarray):
        """ Returns a string to be used as a printable representation
        of a given rectangle."""
        if i in self.neighbours:
            local_indices = numpy.where(self.ghost_owners == i)[0] + self.size_owned
            ghosts = self._index_map.ghosts[self.ghost_owners == i].astype(IntType)
            return ghosts, local_indices.astype(IntType)
        else:
            raise Exception('SubDomain ' + str(i) +
                            ' is not a neighbour  of subdomain ' +
                            str(self.id))
