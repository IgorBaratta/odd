from petsc4py import PETSc


class MatrixContext(object):
    """
    This class gives the Python context for a PETSc Python matrix.
    """
    def __init__(self, Ai: PETSc.Mat, Di: PETSc.Mat):
        """
        Ai is the local matrix, possibly with transmission conditiion
        Di is the partition of unity matrix
        """
        self.Ai = Ai
        self.Di = Di

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
        y.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
