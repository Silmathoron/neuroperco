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
Percolator class to generate and setup the network, then simulate the activity
'''

import logging

import numpy as np
import nngt

from ._frozendict import _frozendict
from ._nprc_log import _log_message
from ._tools import nonstring_container


# get logger

logger = logging.getLogger(__name__)


# Percolator class

class Percolator(object):

    def __init__(self, network=None, neuron_param=None, simu_param=None):
        '''
        Create a new Percolator instance.

        Parameters
        ----------
        network : :class:`nngt.Graph` or subclass, optional (default: None)
            Network on which the activity will be simulated.
        neuron_param : dict, optional (default: None)
            Parameters for the neurons. See
            :func:`neuroperco.Percolator.set_param_neuron` for details.
        sim_param : dict, optional (default: None)
            Parameters for the simulatrion. See
            :func:`neuroperco.Percolator.set_param_simu` for details.
        '''
        self._network      = network
        # neuronal parameters
        self._th     = None
        self._b      = None
        self._v_min  = None
        self._tau_th = None
        self._tau_m  = None
        self._tau_s  = None
        # simulation parameters
        self._timestep = None
        self._th_step  = None
        self._an_step  = None
        # update parameters if necessary
        if neuron_param is not None:
            self.set_param_neuron(**neuron_param)
        if simu_param is not None:
            self.set_param_simu(**simu_param)
        # preparation members
        self._prepared = False  # has `prepare` been called?
        self._ps       = None   # percolation phase space
        # simulation data
        self._times    = []
        self._activity = []
        self._thetas   = []

    @property
    def network(self):
        ''' The network. '''
        return self._network

    @property
    def neuron_param(self):
        '''
        The neuronal parameters.

        Note
        ----
        These are read-only. Use
        :func:`~neuroperco.Percolator.set_param_neuron` to set them.
        '''
        p = {
            "threshold": self._th,
            "b": self._b,
            "tau_th": self._tau_th,
            "tau_m": self._tau_m,
            "tau_s": self._tau_s,
        }
        return p

    @property
    def simu_param(self):
        '''
        The simulation parameters.

        Note
        ----
        These are read-only. Use
        :func:`~neuroperco.Percolator.set_param_simu` to set them.
        '''
        p = {
            "timestep": self._timestep,
            "th_step": self._th_step,
            "active_neurons_step": self._an_step,
        }
        return p

    def load_network(self, filename, **kwargs):
        '''
        Load a network from a file.

        See
        ---
        :func:`nngt.load_from_file`
        '''
        self._network = nngt.load_from_file(filename, **kwargs)
        delays        = self._network.get_delays()
        d0            = delays[0]
        for d in delays:
            if d != d0:
                self._network = None
                raise ValueError("Invalid network: all delays must be equal.")

    def create_network(self, graph_type, population=None, **kwargs):
        '''
        Create a network using NNGT.

        Parameters
        ----------
        graph_type : str
            Name of the graph type, among "all_to_all", "distance_rule",
            "erdos_renyi", "fixed_degree", "gaussian_degree", "newman_watts",
            "random_scale_free".
        population : :class:`nngt.NeuralPop`, optional (default: None)
        kwargs : dict
            Keyword arguments for the graph.

        See
        ---
        :mod:`nngt.generation` for details on the graph models and their
        parameters.
        '''
        kwargs["graph_type"] = graph_type
        kwargs["population"] = population
        if "delays" in kwargs:
            if "distribution" in kwargs["delays"]:
                assert kwargs["delays"]["distribution"] == "constant", \
                    "Only uniform delays are supported"
            else:
                assert isinstance(kwargs["delays"], (int, float)), \
                    "Network delays must be a number"
            
        self._network = nngt.generation.generate(**kwargs)

    def set_param_neuron(self, threshold=None, b=None, v_min=None, tau_th=None,
                         tau_m=None, tau_s=None):
        '''
        Set the neuronal parameters.
        Values set to None are not modified (previous values are kept if
        existing).

        Parameters
        ----------
        threshold : double or array of doubles, optional (default: None)
            Threshold $\theta_0$ above which the neurons switch from resting to
            active.
        b : non-negative double, optional (default: None)
            Value by which the threshold of a neuron is increased after a
            spike. This refers to adaptation mechanisms, making it harder
            to
        v_min : double, optional (default: None)
            Minimal potential of a neuron. The neuron state is described by
            a number $v$ which can vary between $v_{min}$ (usually $< 0$) and
            $\theta$.
        tau_th : double, optional (default: None)
            Typical time with which the neuronal threshold goes back to
            its resting value $theta_0$ after having been increased by `b`
            following a spike.
        tau_m : double, optional (default: None)
            Membrane time constant of the neurons. If zero, neurons are
            considered as responding immediately.
        tau_s : double, optional (default: None)
            Typical time it takes for a synaptic input to reach its peak value.
            If zero, synapses are considered to transfer signals immediately.

        Note
        ----
        The total delay between the activation of a neuron and the reception
        of the signal sent to its out-neighbours is given by
        $D = d + tau_m + tau_s$, where $d$ is the transmission delay along
        the connections.
        '''
        if self._prepared:
            raise RuntimeError(
                "Cannot change parameters after `prepare` has been called.")

        if nonstring_container(threshold):
            self._th = threshold
        elif threshold is not None:
            n = self._network.node_nb()
            self._th = np.full(n, threshold)

        if nonstring_container(b):
            raise ValueError("`b` should be the same for all neurons.")
        elif b is not None:
            assert b >= 0, "`b` must be non-negative."
            self._b = b

        if nonstring_container(v_min):
            raise ValueError("`v_min` should be the same for all neurons.")
        elif v_min is not None:
            self._v_min = v_min

        if nonstring_container(tau_th):
            raise ValueError("`tau_th` should be the same for all neurons.")
        elif tau_th is not None:
            self._tau_th = tau_th

        if nonstring_container(tau_m):
            raise ValueError("`tau_m` should be the same for all neurons.")
        elif tau_m is not None:
            if self._timestep is not None:
                # check that the parameters are compatible with the timestep
                try:
                    d = self._network._d["value"]
                except:
                    d = self._network.get_delays()[0]
                tau_s = self._tau_s if tau_s is None else tau_s
                t     = self._timestep

                if None not in (d, tau_m, tau_s):
                    assert np.isclose(int(t/(d + tau_m + tau_s)),
                                  t/(d + tau_m + tau_s), 0.1), \
                    "`timestep` must be a multiple of (d + tau_m + tau_s)."
            # set the value if valid
            self._tau_m = tau_m

        if nonstring_container(tau_s):
            raise ValueError("`tau_s` should be the same for all neurons.")
        elif tau_s is not None:
            if self._timestep is not None:
                # check that the parameters are compatible with the timestep
                try:
                    d = self._network._d["value"]
                except:
                    d = self._network.get_delays()[0]
                tau_m = self._tau_m if tau_m is None else tau_m
                t     = self._timestep

                if None not in (d, tau_m, tau_s):
                    assert np.isclose(int(t/(d + tau_m + tau_s)),
                                      t/(d + tau_m + tau_s), 0.1), \
                        "`timestep` must be a multiple of (d + tau_m + tau_s)."
            # set the value if valid
            self._tau_s = tau_s

    def set_param_simu(self, timestep=None, th_step=None,
                       active_neurons_step=None):
        '''
        Set the simulation parameters.
        Values set to None are not modified (previous values are kept if
        existing).

        Parameters
        ----------
        timestep : non-negative double 
            Timestep of the simulation. This must be a multiple of the total
            delay in the network (D = d + tau_m + tau_s).
        th_step : stricly positive double, optional (default: None)
            Precision with which the response of the network to a given
            stimulation will be interpolated, depending on the neurons'
            potential. The neuron state is described by a potential $v$ which
            can vary between $v_{min} < 0$ and $\theta$ (the threshold). Before
            a run, we compute the percolation response for possible
            $dv = \theta - v$, between $k_{max} - v_{min}$ and $0$, with a step
            `th_step` ($k_{max} is the maximum in-degree in the network).
            Increasing `th_step` reduces the precision of the prediction but
            also increases speed and diminishes memory consumption.
        active_neurons_step : strictly positive int, optional (default: None)
            In addition to the percolation responses for possible $dv$, we also
            compute the response depending on how many neurons are active,
            between 0 and $N$ (the number of neurons in the network) with a
            step `active_neurons_step`. As for `th_step`, increasing it reduces
            the precision but improves speed and reduces memory consumption.
        '''
        if self._prepared:
            raise RuntimeError(
                "Cannot change parameters after `prepare` has been called.")

        if timestep is not None:
            assert timestep >= 0, "`timestep` must be non-negative."
            tau_m = self._tau_m
            tau_s = self._tau_s
            try:
                d = self._network._d["value"]
            except:
                d = self._network.get_delays()[0]
            if None not in (d, tau_m, tau_s):
                assert np.isclose(int(timestep/(d + tau_m + tau_s)),
                                  timestep/(d + tau_m + tau_s), 0.1), \
                    "`timestep` must be a multiple of (d + tau_m + tau_s)."
            self._timestep = timestep

        if th_step is not None:
            assert th_step > 0, "`th_step` must be strictly positive"
            self._th_step = th_step

        if active_neurons_step is not None:
            assert active_neurons_step > 0, \
                "`active_neurons_step` must be strictly positive"
            self._an_step = active_neurons_step

    def prepare(self):
        '''
        Compute the percolation phase-space that will be used during the run.
        '''
        compute_phase_space(
            network=self._network, simu_param=self.simu_param,
            neuron_param=self.neuron_param, axes_lim=None)
        self._prepare = True

    def run(self, duration):
        '''
        Simulate the activity.

        Parameters
        ----------
        duration : non-negative double
            Simulation time, must be a multiple of `timestep`.
        '''
        pass

    @property
    def activity(self):
        '''
        Return the activity as a (M, 3) array, where M is the number of
        simulated timesteps. First column is the times, second column is the
        number of active neurons at that time, column is the value of
        the average threshold.
        '''
        return np.array((self._times, self._activity, self._thetas)).T
