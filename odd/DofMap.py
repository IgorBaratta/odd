from dolfin.function import functionspace
from petsc4py.PETSc import IntType


class DofMap():
    def __init__(self,
                 V: functionspace.FunctionSpace):

        self._dofmap = V.dofmap
        self._index_map = V.dofmap.index_map

    @property
    def indices(self):
        return self._index_map.indices(True).astype(IntType)

    @property
    def owned_indices(self):
        return self.indices[:self.size_local]

    @property
    def ovl_indices(self):
        return self.indices[self.size_local:]

    @property
    def size_local(self):
        return self._index_map.size_local + self._index_map.num_ghosts

    @property
    def size_owned(self):
        return self._index_map.size_local

    @property
    def size_global(self):
        return self._index_map.size_global
