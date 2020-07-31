# Copyright (C) 2020 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import abc
from enum import Enum

from numpy.core.multiarray import ndarray
from odd.core._index_map import IndexMap


class InsertMode(Enum):
    INSERT = 1
    ADD = 2


class VectorScatter(object, metaclass=abc.ABCMeta):
    """
    Manage communication of data between vectors in parallel.
    """

    def __init__(self, index_map: IndexMap):
        self._initialized = False
        super(VectorScatter, self).__init__()
        self.index_map = index_map

    @abc.abstractmethod
    def forward(self, array: ndarray, insert_mode: InsertMode) -> None:
        pass

    @abc.abstractmethod
    def reverse(self, array: ndarray, insert_mode: InsertMode) -> None:
        pass
