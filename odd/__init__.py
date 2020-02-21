# Copyright (C) 2019 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Main module for odd"""
# flake8: noqa

# Initialise PETSc/MPI
from dolfinx import cpp
import sys
cpp.common.SubSystemsManager.init_logging(sys.argv)
del sys
cpp.common.SubSystemsManager.init_petsc()
del cpp

# Import public interface
from .dofmap import DofMap
from .schwarz import AdditiveSchwarz, SMType
from .subdomain import SubDomainData
from .vector_scatter import PETScVectorScatter, RMAVectorScatter, ScatterType
from .assemble_vector import assemble_vector
from .assemble_matrix import (assemble_matrix, create_matrix,
                              sparsity_pattern, apply_transmission_condition)
from .dirichlet_bc import apply_bc
from .matrix_context import MatrixContext
