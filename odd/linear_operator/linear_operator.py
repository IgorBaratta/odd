# Copyright (C) 2020 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from scipy import sparse
from odd.index_map import IndexMap

from numpy.core.multiarray import ndarray


class LinearOperator(object):
    r"""
    Base Class for Linear Operator
    The parallel layout of the linear operator is handle by the IndexMap.
    """

    def __init__(self, local_mat: sparse.spmatrix, index_map: IndexMap):
        """
        Parameters
        ----------

        """
        # Check if sizes match the parallel layout:
        self.check_sizes(local_mat, index_map)

        self._local_mat = local_mat
        self._index_map = index_map

    def apply(self, x: ndarray) -> ndarray:
        y: ndarray = self._local_mat @ x
        return y

    @staticmethod
    def check_sizes(local_mat: sparse.spmatrix, index_map: IndexMap) -> bool:
        return True
