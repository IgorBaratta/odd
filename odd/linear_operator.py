from scipy import sparse as sp
from .indexmap import IndexMap


class LinearOperator(object):
    """
    Base Class for Linear Operator
    """
    def __init__(self, Ai: sp.spmatrix, Di: sp.spmatrix, indexmap: IndexMap):
        """
        Ai is the local matrix, possibly with transmission conditiion
        Di is the partition of unity matrix
        """
        self.Ai = Ai
        self.indexmap = indexmap

    def mult(self, mat, x, y):
        """
        Computes the matrix-vector product, y = Ax.
        """
        with x.localForm() as x_local:
            if not x_local:
                raise RuntimeError('X vector is not ghosted')
            with y.localForm() as y_local:
                if not y_local:
                    raise RuntimeError('X vector is not ghosted')
                self.Ai.mult(x_local, y_local)
        # y.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
