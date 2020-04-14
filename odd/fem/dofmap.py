# Copyright (C) 2020 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""
Python wrapper for dolfinx DofMap.
We only keep native and numpy objects.
"""

import numpy
import collections
import dolfinx.cpp.fem

DofMapWrapper = collections.namedtuple("DofMapWrapper", "dof_array num_cell_dofs size")


def dofmap_wrapper(dofmap: dolfinx.cpp.fem.DofMap):
    num_cell_dofs = dofmap.dof_layout.num_dofs
    dof_array = dofmap.list().array()
    size = numpy.max(dof_array) + 1
    return DofMapWrapper(dof_array, num_cell_dofs, size)
