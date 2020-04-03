# Copyright (C) 2020 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Main module for odd"""
# flake8: noqa

# Import public interface
from .dofmap import DofMap
from .schwarz import AdditiveSchwarz, SMType
from .subdomain import SubDomainData
from .vector_scatter import PETScVectorScatter, RMAVectorScatter, ScatterType
from .assemble_vector import assemble_vector
from .assemble_matrix import assemble_matrix, apply_transmission_condition
from .dirichlet_bc import apply_bc
from .matrix_context import MatrixContext
