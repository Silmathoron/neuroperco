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

'''
Computing the percolation phase space
'''

import logging
import itertools

import numpy as np

from ._nprc_log import _log_message
from ._tools import nonstring_container


# get logger

logger = logging.getLogger(__name__)


# compute phase space

def compute_phase_space(network, neuron_param, simu_param,
                        axes=('threshold', 'active_neurons'),
                        axes_limits=None, num_timesteps=1, num_trials=100):
    '''
    Compute the phase space of the percolation phenomenon for a specific
    network, neuron, and simulation parameters.

    Parameters
    ----------
    network : :class:`nngt.Graph` or subclass
        Network on which the activity will be simulated.
    neuron_param : dict
        Neuronal parameters (see :class:`~neuroperco.Percolator`)
    simu_param : dict
        Simulation parameters (see :class:`~neuroperco.Percolator`)
    axes : list, optional (default: ['threshold', 'active_neurons'])
        Axes of phase-space along which the behavior will be tested. At least
        one entry is required.
    axes_limits : dict, optional (default: automatically computed)
        Limits of the `axes`, e.g.
        ``{'threshold': (0., 10.), 'active_neurons': (0, 100)}``.
    num_timesteps : int, optional (default: 1)
        Number of timesteps, i.e. of spike propagation and integrations, that
        will be used to compute the final number of active neurons after the
        percolation process.
    num_trials : int, optional (default: 100)
        Number of trials that will be performed for each point of phase-space.

    Return
    ------
    ps : array of dimension len(axes)
        Final number of active neurons after `num_timesteps`, for each
        combination of the axes values.
    '''
    threshold, b, v_min, tau_th, tau_m, tau_s = _check_neuron_params(
        network, timestep, neuron_param,simu_param)
    th_step, an_step = _check_simu_params(
        network, timestep, tau_m, tau_s, simu_param)

    # last checks + prepare phase space
    dimensions  = []
    axes_values = {}
    axes_size   = [0, 0]

    if 'active_neurons' in axes:
        assert an_step is not None, "`active_neurons` axis required but no " +\
            "`active_neurons_step` provided in `simu_params`."
        if axes_limits is not None:
            an_min, an_max = axes_limits["active_neurons"]
        else:
            an_min, an_max = 0, network.node_nb()
        num_an_steps = (an_max - an_min) / an_step
        axes_values['active_neurons'] = np.arange(
            an_min, an_max + an_step, an_step, dtype=int)
        dimensions.append(num_an_steps)
        axes_size[axes.index('active_neurons')] = num_an_steps

    if 'threshold' in axes:
        assert dv_step is not None, "`threshold` axis required but no " +\
            "`th_step` provided in `simu_params`."
        if axes_limits is not None:
            th_min, th_max = axes_limits["threshold"]
        else:
            k_max = np.max(network.get_degrees('in'))
            s_max = np.max(network.get_weights())
            th_min, th_max = 0., k_max*s_max
        num_th_steps = (th_max - th_min) / th_step
        axes_values['threshold'] = np.arange(th_min, th_max + th_step, th_step)
        dimensions.append(num_th_steps)
        axes_size[axes.index('threshold')] = num_an_steps

    ps = np.zeros(dimension)

    if len(axes) == 2:
        xs = itertools.repeat(enumerate(axes_values[axes[0]), axes_size[1])
        ys = (itertools.repeat((i,y), axes_size[0])
              for i, y in enumerate(axes_values[axes[1]))
        coords = itertools.izip(xs, ys)
    else:
        coords = enumerate(next(iter(axes_values.values())))

    # compute phase space
    for idx, val in coords:
        ps[idx]


def _check_neuron_params(network, timestep, neuron_param, simu_param):
    params = []

    if nonstring_container(threshold):
        params.append(neuron_param["threshold"])
    elif neuron_param["threshold"] is not None:
        n = network.node_nb()
        params.append(np.full(n, neuron_param["threshold"]))

    if nonstring_container(neuron_param["b"]):
        raise ValueError("`b` should be the same for all neurons.")
    elif neuron_param["b"] is not None:
        assert neuron_param["b"] >= 0, "`b` must be non-negative."
        params.append(neuron_param["b"])

    if nonstring_container(neuron_param["v_min"]):
        raise ValueError("`v_min` should be the same for all neurons.")
    elif neuron_param["v_min"] is not None:
        params.append(neuron_param["v_min"])

    if nonstring_container(neuron_param["tau_th"]):
        raise ValueError("`tau_th` should be the same for all neurons.")
    elif neuron_param["tau_th"] is not None:
        params.append(neuron_param["tau_th"])

    if nonstring_container(neuron_param["tau_m"]):
        raise ValueError("`tau_m` should be the same for all neurons.")
    elif neuron_param["tau_m"] is not None:
        # check that the parameters are compatible with the timestep
        try:
            d = network._d["value"]
        except:
            d = network.get_delays()[0]

        if None not in (d, neuron_param["tau_m"], neuron_param["tau_s"]):
            assert np.isclose(
                int(timestep/(d + neuron_param["tau_m"] +
                    neuron_param["tau_s"])),
                timestep/(d + neuron_param["tau_m"] +
                    neuron_param["tau_s"]),
                0.1), "`timestep` must be a multiple of (d + tau_m + tau_s)."
        # set the value if valid
        params.append(neuron_param["tau_m"])

    if nonstring_container(neuron_param["tau_s"]):
        raise ValueError("`tau_s` should be the same for all neurons.")
    elif neuron_param["tau_s"] is not None:
        # check that the parameters are compatible with the timestep
        try:
            d = network._d["value"]
        except:
            d = network.get_delays()[0]

        if None not in (d, neuron_param["tau_m"], neuron_param["tau_s"]):
            assert np.isclose(
                int(timestep/(d + neuron_param["tau_m"] +
                    neuron_param["tau_s"])),
                timestep/(d + neuron_param["tau_m"] +
                    neuron_param["tau_s"]),
                0.1), "`timestep` must be a multiple of (d + tau_m + tau_s)."
        # set the value if valid
        params.append(neuron_param["tau_s"])

    return params


def _check_simu_params(network, timestep, tau_m, tau_s, simu_param):
    params = []

    assert timestep >= 0, "`timestep` must be non-negative."
    try:
        d = network._d["value"]
    except:
        d = network.get_delays()[0]
    if None not in (d, tau_m, tau_s):
        assert np.isclose(int(timestep/(d + tau_m + tau_s)),
                          timestep/(d + tau_m + tau_s), 0.1), \
            "`timestep` must be a multiple of (d + tau_m + tau_s)."

    if simu_param.get("dv_step", None) is not None:
        assert simu_param["dv_step"] > 0, "`dv_step` must be strictly positive"
    params.append(simu_param.get("dv_step", None))

    if simu_param.get("active_neurons_step", None) is not None:
        assert simu_param["active_neurons_step"] > 0, \
            "`active_neurons_step` must be strictly positive"
    params.append(simu_param.get("active_neurons_step", None))

    return params
