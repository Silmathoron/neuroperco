#!/usr/bin/env python
#-*- coding:utf-8 -*-
#
# This file is part of the neuroperco project, which aims at providing tools to
# easily generate activity of neuronal network thourhg a dynamical percolation
# framework.
# Copyright (C) 2018 SENeC Initiative
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
NeuroPerco
"""

import logging

from .percolator import Percolator
from ._nprc_log import _log_message


__version__ = "0.1.0"

_logger = logging.getLogger(__name__)


