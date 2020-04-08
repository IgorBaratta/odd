# Copyright (C) 2020 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Main module for odd"""
# flake8: noqa

__all__ = ["assemble_matrix", "assemble_matrix", "apply_bc"]

from .assemble_matrix import assemble_matrix
from .assemble_vector import assemble_vector
from .dirichlet_bc import apply_bc
