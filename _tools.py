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
# along with this program. If not, see <http://www.gnu.org/licenses/>.

""" Test functions for the NNGT """

import tarfile
import collections
try:
    from collections.abc import Container as _container
except:
    from collections import Container as _container

import numpy as np

import nngt


def on_master_process():
    '''
    Check whether the current code is executing on the master process (rank 0)
    if MPI is used.

    Returns
    -------
    True if rank is 0, if mpi4py is not present or if MPI is not used,
    otherwise False.
    '''
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if rank == 0:
            return True
        else:
            return False
    except ImportError:
        return True


def mpi_checker(logging=False):
    '''
    Decorator used to check for mpi and make sure only rank zero is used
    to store and generate the graph if the mpi algorithms are activated.
    '''
    def decorator(func):
        def wrapper(*args, **kwargs):
            # when using MPI, make sure everyone waits for the others
            try:
                from mpi4py import MPI
                comm = MPI.COMM_WORLD
                comm.Barrier()
            except ImportError:
                pass
            # check backend ("nngt" is fully parallel, not the others)
            backend = False
            if not logging:
                backend = nngt.get_config("backend") == "nngt"
            if backend or on_master_process():
                return func(*args, **kwargs)
            else:
                return None
        return wrapper
    return decorator


def nonstring_container(obj):
    '''
    Returns true for any iterable which is not a string or byte sequence.
    '''
    if not isinstance(obj, _container):
        return False
    try:
        if isinstance(obj, unicode):
            return False
    except NameError:
        pass
    if isinstance(obj, bytes):
        return False
    if isinstance(obj, str):
        return False
    return True


def is_integer(obj):
    return isinstance(obj, (int, np.integer))


def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))
