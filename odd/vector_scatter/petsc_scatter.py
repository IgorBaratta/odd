from petsc4py import PETSc
from numpy.core.multiarray import ndarray

from .vector_scatter import VectorScatter, InsertMode
from ..index_map import IndexMap


class PETScVectorScatter(VectorScatter):
    """
    Fallback to PETSc vector scatter
    """
    def __init__(self, index_map: IndexMap):
        """

        Parameters
        ----------
        index_map : IndexMap
        """
        super().__init__(index_map)

    def forward(self, array: ndarray, insert_mode: InsertMode = InsertMode.ADD):
        ghosts = self.index_map.ghosts.astype(PETSc.IntType)
        vec = PETSc.Vec().createGhostWithArray(ghosts, array)
        vec.ghostUpdate(InsertMode.ADD.value, PETSc.ScatterMode.REVERSE)
        with vec.localForm() as vec_local:
            array[:] = vec_local.array.astype(array.dtype)

    def reverse(self, array: ndarray, insert_mode: InsertMode = InsertMode.INSERT):
        ghosts = self.index_map.ghosts.astype(PETSc.IntType)
        vec = PETSc.Vec().createGhostWithArray(ghosts, array)
        vec.ghostUpdate(insert_mode.value, PETSc.ScatterMode.FORWARD)
        with vec.localForm() as vec_local:
            array[:] = vec_local.array.astype(array.dtype)

    @property
    def initialized(self):
        return self._initialized
