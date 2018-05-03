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
from collections import deque

import numpy as np

from ._nprc_log import _log_message
from ._tools import nonstring_container


# get logger

logger = logging.getLogger(__name__)


# compute phase space

class PhaseSpace(object):

    def __init__(self, network, neuron_param, simu_param):
        self.network = network
        # check parameters
        _check_neuron_params(network, neuron_param, simu_param)
        _check_simu_params(network, neuron_param, simu_param)
        # neuronal parameters
        self.th     = neuron_param["threshold"]
        self.b      = neuron_param["b"]
        self.v_min  = neuron_param["v_min"]
        self.tau_th = neuron_param["tau_th"]
        self.tau_m  = neuron_param["tau_m"]
        self.tau_s  = neuron_param["tau_s"]
        # simulation parameters
        self.timestep = simu_param["timestep"]
        self.th_step  = simu_param["th_step"]
        self.an_step  = simu_param["active_neurons_step"]

    def compute_phase_space(self, axes=('threshold', 'active_neurons'),
                            axes_limits=None, num_timesteps=1, num_trials=100,
                            num_initially_active=None):
        '''
        Compute the phase space of the percolation phenomenon for a specific
        network, neuron, and simulation parameters.

        Parameters
        ----------
        axes : list, optional (default: ['threshold', 'active_neurons'])
            Axes of phase-space along which the behavior will be tested. At
            least one entry is required.
        axes_limits : dict, optional (default: automatically computed)
            Limits of the `axes`, e.g.
            ``{'threshold': (0., 10.), 'active_neurons': (0, 100)}``.
        num_timesteps : int, optional (default: 1)
            Number of timesteps, i.e. of spike propagation and integrations,
            that will be used to compute the final number of active neurons
            after the percolation process.
        num_trials : int, optional (default: 100)
            Number of trials that will be performed for each point of
            phase-space.
        num_initially_active : int, optional (default: None)
            Number of initially active neurons. Required when only the
            threshold is varied.

        Return
        ------
        ps : array of dimension len(axes)
            Final number of active neurons after `num_timesteps`, for each
            combination of the axes values.
        '''
        # last checks + prepare phase space
        dimensions  = []  # first active_neurons, then threshold if both
        axes_limits = {} if axes_limits is None else axes_limits
        axes_values = {}
        axes_size   = [0, 0]

        if 'active_neurons' in axes:
            assert self.an_step is not None, "`active_neurons` axis " +\
                "required but no `active_neurons_step` provided in " +\
                "`simu_params`."
            an_min, an_max = axes_limits.get(
                "active_neurons", [0, self.network.node_nb()])
            axes_values['active_neurons'] = np.arange(
                an_min, an_max + self.an_step, self.an_step, dtype=int)
            num_an_steps = len(axes_values['active_neurons'])
            dimensions.append(num_an_steps)
            axes_size[axes.index('active_neurons')] = num_an_steps

        if 'threshold' in axes:
            assert self.th_step is not None, "`threshold` axis required " +\
                "but no `th_step` provided in `simu_params`."
            k_max = np.max(network.get_degrees('in'))
            s_max = np.max(network.get_weights())
            th_min, th_max = axes_limits.get("threshold", [0, k_max*s_max])
            axes_values['threshold'] = np.arange(
                th_min, th_max + self.th_step, self.th_step)
            num_th_steps = len(axes_values)
            dimensions.append(num_th_steps)
            axes_size[axes.index('threshold')] = num_th_steps
            # if only threshold in axes, check that `active_neurons` is defined
            if 'active_neurons' not in axes:
                assert num_initially_active is not None, \
                    "`num_initially_active` must  be provided as a default " +\
                    "value when varying only `threshold`."

        ps = np.zeros(dimensions)

        # compute phase space
        if len(axes) == 2:
            ans = itertools.repeat(
                enumerate(axes_values['active_neurons'], axes_size[1]))
            ths = (itertools.repeat((i, y), axes_size[0])
                  for i, y in enumerate(axes_values['threshold']))
            # iterate
            for k in range(num_trials):
                for (i, an), (j, th) in itertools.izip(ans, ths):
                    ps[i, j] += self.percolation(
                        num_timesteps, num_initially_active=an, threshold=th)
        elif 'active_neurons' in axes:
            for k in range(num_trials):
                for i, an in enumerate(axes_values['active_neurons']):
                    ps[i] += self.percolation(
                        num_timesteps, num_initially_active=an)
        else:
            for k in range(num_trials):
                for i, th in enumerate(axes_values['threshold']):
                    ps[i] += self.percolation(
                        num_timesteps, threshold=th,
                        num_initially_active=num_initially_active)
        ps /= num_trials

        return axes_values, ps

    def percolation(self, num_timesteps, num_initially_active, threshold=None,
                    show=False):
        '''
        Run a percolation process

        Parameters
        ----------
        num_timesteps : int
            Number of timesteps, i.e. of spike propagation and integrations,
            that will be used to compute the final number of active neurons
            after the percolation process.
        num_initially_active : float
            Number of initially active neurons.
        threshold : float, optional (default: in `neuron_param`)
            Threshold of the neurons.
        show : bool, optional (default: False)
            Wether the graph of the percolation event should be shown.

        Return
        ------
        final_active : int
            Final number of active neurons.
        '''
        # set the params
        th = self.th if threshold is None else threshold
        final_active = num_initially_active
        t = 0
        active = deque(
            np.random.choice(
                self.network.node_nb(), num_initially_active, replace=False))
        v_m = np.full(self.network.node_nb(), 0.)
        excitable = np.full(self.network.node_nb(), 1)
        excitable[active] = 0

        # create activation time node property in the network
        at = np.full(self.network.node_nb(), -1)
        at[active] = 0
        if "activation_time" not in self.network.nodes_attributes:
            self.network.new_node_attribute(
                "activation_time", "int", values=at)
        else:
            self.network.set_node_attribute("activation_time", values=at)

        # run the percolation event
        while t < num_timesteps:
            new_active = []
            while active:
                # get first active neuron
                n = active.popleft()
                # get neighbours and edges
                nn    = list(self.network.neighbours(n, "out"))
                edges = np.array((np.repeat(n, len(nn)), nn), dtype=int).T
                # update neuronal properties
                v_m[nn] += self.network.get_weights(edges)
                over_th  = np.where((v_m > th) * excitable)[0]
                excitable[over_th] = 0  # cannot be activated several times
                # update activity
                final_active += len(over_th)
                new_active.extend(over_th)
            t += 1
            active.extend(new_active)  # add new active neurons
            self.network.set_node_attribute(
                "activation_time", val=t, nodes=new_active)

        if show:
            from nngt.plot import draw_network
            draw_network(self.network, ncolor="activation_time",
                         decimate=-1, show=True)
            # ~ draw_network(self.network, decimate=-1, show=True)

        return final_active


# tool functions

def _check_neuron_params(network, neuron_param, simu_param):
    timestep = simu_param["timestep"]

    if nonstring_container(neuron_param["b"]):
        raise ValueError("`b` should be the same for all neurons.")
    elif neuron_param["b"] is not None:
        assert neuron_param["b"] >= 0, "`b` must be non-negative."

    if nonstring_container(neuron_param["v_min"]):
        raise ValueError("`v_min` should be the same for all neurons.")

    if nonstring_container(neuron_param["tau_th"]):
        raise ValueError("`tau_th` should be the same for all neurons.")

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


def _check_simu_params(network, neuron_param, simu_param):
    timestep = simu_param["timestep"]
    tau_m, tau_s = neuron_param["tau_m"], neuron_param["tau_s"]
    
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

    if simu_param.get("active_neurons_step", None) is not None:
        assert simu_param["active_neurons_step"] > 0, \
            "`active_neurons_step` must be strictly positive"
