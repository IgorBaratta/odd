# Copyright (C) 2020 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
from scipy import sparse
from odd.index_map import IndexMap

from numpy.core.multiarray import ndarray


class LinearOperator(sparse.csr_matrix):
    r"""
    A sparse distributed matrix.
    The parallel layout of the linear operator is handled by the IndexMap.
    """

    def __init__(self, local_mat: sparse.spmatrix, index_map: IndexMap):
        sparse.csr_matrix.__init__(self, local_mat, shape=None, dtype=None, copy=False)
        self.index_map = index_map

    def apply(self, x: ndarray) -> ndarray:
        return self.__matmul__(x)

    def __call__(self, x: ndarray) -> ndarray:
        return self.apply(x)
    #
    # def count_nonzero(self):
    #     pass
    #
    # def resize(self, shape):
    #     pass
    #
    # @staticmethod
    # def check_sizes(local_mat: sparse.spmatrix, index_map: IndexMap) -> bool:
    #     rows, cols = local_mat.shape
    #     sizes = [index_map.local_size, index_map.owned_size]
    #     if (rows in sizes) and (cols in sizes):
    #         return True
    #     else:
    #         print("The local matrix size and parallel layout do not match.")
    #         return False

