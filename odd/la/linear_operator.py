# Copyright (C) 2020 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
import numpy
from scipy import sparse
from odd.index_map import IndexMap
from .vector import Vector


class LinearOperator(object):
    r"""
    A sparse distributed matrix.
    The parallel layout of the linear operator is handled by the IndexMap.
    """

    def __init__(self, local_mat: sparse.spmatrix, row_map: IndexMap):
        self.row_map = row_map
        self.local_mat = local_mat
        assert self.check_sizes(local_mat, row_map)

    def __call__(self, vector: Vector):
        c = vector.copy()
        c[:] = self.local_mat * vector._array
        return c


    @staticmethod
    def check_sizes(local_mat: sparse.spmatrix, index_map: IndexMap) -> bool:
        rows, cols = local_mat.shape
        sizes = [index_map.local_size, index_map.owned_size]
        if (rows in sizes) and (cols in sizes):
            return True
        else:
            print("The local matrix size and parallel layout do not match.")
            return False
