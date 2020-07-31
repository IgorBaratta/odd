# Copyright (C) 2020 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Main module for odd"""
# flake8: noqa

# Import public interface
from .core import DistArray, empty, full, ones, zeros
from .communication import parallel_reduce

