# Copyright (C) 2020 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Main module for odd"""
# flake8: noqa

__all__ = ['IndexMap', 'AdditiveSchwarz', 'SMType', 'SubDomainData',
           'PETScVectorScatter', 'RMAVectorScatter', 'ScatterType',
           'LinearOperator', 'fem']

# Import public interface
from .indexmap import IndexMap
from .schwarz import AdditiveSchwarz, SMType
from .subdomain import SubDomainData
from .vector_scatter import PETScVectorScatter, RMAVectorScatter, ScatterType
from .linear_operator import LinearOperator

from odd import fem
