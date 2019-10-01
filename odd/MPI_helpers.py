from petsc4py import PETSc


def vector_scatter_fwd(self, vector):
    self.scatter_l2g(self.workl_2, vector,
                     PETSc.InsertMode.ADD_VALUES)


def vector_scatter_rev(self, vector):
    self.scatter_l2g(vector, self.vec1,
                     PETSc.InsertMode.INSERT_VALUES,
                     PETSc.ScatterMode.SCATTER_REVERSE)
