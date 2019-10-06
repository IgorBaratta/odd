from dolfin.function import functionspace
from petsc4py.PETSc import IntType
import numpy


class DofMap:
    """ TODO : class docstring """
    def __init__(self,
                 V: functionspace.FunctionSpace):

        self._dofmap = V.dofmap
        self._index_map = V.dofmap.index_map
        self._id = V.mesh.mpi_comm().rank

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
        return self.indices[self.size_local:]

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
        return self._index_map.ghosts

    def neighbour_ghosts(self, i: int) -> numpy.ndarray:
        """ Returns a string to be used as a printable representation
        of a given rectangle."""
        if i in self.neighbours:
            return self._index_map.ghosts[self.ghost_owners == i]
        else:
            raise Exception('SubDomain ' + str(i) +
                            ' is not a neighbour  of subdomain ' +
                            str(self._id))
