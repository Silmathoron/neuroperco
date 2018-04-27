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

import os, errno
from setuptools import setup, find_packages


# create directory
directory = 'neuroperco/'
try:
    os.makedirs(directory)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise


# move important 
move = []

for path, subdirs, files in os.walk("."):
    for subdir in subdirs:
        try:
            os.makedirs(directory + path[2:] + "/" + subdir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    for f in files:
        if f != "setup.py" and f.endswith(".py"):
            move.append(path[2:] + "/" + f)

for f in move:
    os.rename(f, directory + f)


from neuroperco import __version__


try:
    # install
    setup(
        name = 'neuroperco',
        version = __version__,
        description = 'Python module to simulate network activity through '
                      'dynamical percolation.',
        package_dir = {'': '.'},
        packages = find_packages('.'),

        # Requirements
        install_requires = ['numpy', 'scipy>=0.11', 'matplotlib', 'nngt'],

        # Metadata
        url = 'https://github.com/Silmathoron/neuroperco',
        author = 'Tanguy Fardet',
        author_email = 'tanguy.fardet@univ-paris-diderot.fr',
        license = 'GPL3',
        keywords = 'dynamical percolation neuronal networks activity'
    )
finally:
    for fname in move:
        os.rename(directory + fname, fname)
