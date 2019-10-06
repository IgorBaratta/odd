from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import InsertMode
from petsc4py.PETSc import ScatterMode
import numpy
from .DofMap import DofMap


class VectorScatter():
    """ TODO : class docstring """
    def __init__(self, dofmap: DofMap, comm: MPI.Intracomm):
        self.dofmap = dofmap
        self.comm = comm

    def forward(self):
        pass

    def reverse(self):
        pass


class PETScVectorScatter(VectorScatter):
    """ TODO : class docstring """
    def __init__(self, dofmap: DofMap,
                 local_vec: PETSc.Vec,
                 global_vec: PETSc.Vec,
                 comm: MPI.Intracomm):
        super().__init__(dofmap, comm)

        self._is_local = PETSc.IS().createGeneral(self.dofmap.indices)
        self._vec_scatter = PETSc.Scatter()
        self._vec_scatter.create(local_vec, None, global_vec, self._is_local)

    def forward(self, local_vec: PETSc.Vec, global_vec: PETSc.Vec):
        """ TODO : Add descrtiption """
        self._vec_scatter(local_vec, global_vec,
                          InsertMode.ADD_VALUES,
                          ScatterMode.SCATTER_FORWARD)

    def reverse(self, local_vec: PETSc.Vec, global_vec: PETSc.Vec):
        """ TODO : Add descrtiption """
        self._vec_scatter(global_vec, local_vec,
                          InsertMode.INSERT_VALUES,
                          ScatterMode.SCATTER_REVERSE)


class RMAVectorScatter(VectorScatter):
    """ TODO : class docstring """
    def scatter_forward(self, local_data: numpy.ndarray, remote_data: numpy.ndarray):

        # Open window into owned data
        window = MPI.Win.Create(local_data, comm=self.comm)
        window.Fence()

    def scatter_reverse(self, local_data: numpy.ndarray, remote_data: numpy.ndarray):
        """ TODO : Add descrtiption """
        # Open window into owned data
        window = MPI.Win.Create(local_data, comm=self.comm)
        window.Fence()
