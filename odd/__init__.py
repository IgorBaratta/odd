# Copyright (C) 2020 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Main module for odd"""
# flake8: noqa

# Import public interface
from .index_map import IndexMap
from odd.linear_operator import LinearOperator
from odd.preconditioner.schwarz import AdditiveSchwarz, SMType
from odd.vector_scatter import VectorScatter, PETScVectorScatter

from odd import fem

__all__ = ['IndexMap',
           'AdditiveSchwarz', 'SMType',
           'VectorScatter', 'PETScVectorScatter',
           'LinearOperator',
           'fem']
