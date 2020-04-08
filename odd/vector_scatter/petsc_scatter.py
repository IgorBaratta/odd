from petsc4py import PETSc

from numpy.core.multiarray import ndarray

from .vector_scatter import VectorScatter, InsertMode
from ..index_map import IndexMap


class PETScVectorScatter(VectorScatter):
    def __init__(self, index_map: IndexMap):
        super().__init__(index_map)
        self._is_local = PETSc.IS().createGeneral(self._index_map.indices)
        self.vec = PETSc.Vec().create()
        self.vec.setType(PETSc.Vec.Type.MPI)
        self.vec.setSizes((index_map.owned_size, None))
        self.vec.setMPIGhost(index_map.ghosts)
        self._initialized = True

    def forward(self, array: ndarray, insert_mode: InsertMode = InsertMode.INSERT):
        self.vec.setArray(array)
        self.vec.ghostUpdate(insert_mode, PETSc.ScatterMode.FORWARD)

    def reverse(self, array: ndarray, insert_mode: InsertMode = InsertMode.ADD):
        self.vec.setArray(array)
        self.vec.ghostUpdate(insert_mode, PETSc.ScatterMode.REVERSE)

    @property
    def initialized(self):
        return self._initialized
