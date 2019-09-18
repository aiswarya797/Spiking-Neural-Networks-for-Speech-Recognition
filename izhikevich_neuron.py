#-*- coding: utf-8 -*-
## Izhikevich Neuron
## Introduction to izhikevich neurons 
"""
Izhikevich’s model is part of the general representation category so while it does model biological behaviours,the parameters have no biological basis.The LIF model is part of the generic threshold fire category so it is not biologically realistic in anyway except that it can integrate input and fire at some threshold.
"""

import numpy as np
from matplotlib.pyplot import plot
import Synapse

class izhikevich:

    def __init__(self):
        # pre_syn and post_syn are the pre and post synapse times. synapses array hold the list of synapses associated with a neuron

        self.pre_syn = []
        self.post_syn = []
        self.synapses = []
        self.v_rest = -60
        self.v_threshold = -40
        self.v_peak = 35
        self.C = 100
        self.K = 0.7
        self.a=0.03
        self.b = -2
        self.c = -50
        self.d = 100


    def neuron_simulation(self, time_ita, current):
        # a,b,c,d parameters for Izhikevich model
        # time_ita time iterations for euler method
        # current list of current for each time step
        
        spike_times = []
        v = self.c     #v_init
        u = v * self.b
        v_plt = np.zeros(time_ita)
        u_plt = np.zeros(time_ita)
        spike = np.zeros(time_ita)
        num_spikes = 0
        tstep = 0.1  # ms
        ita = 0
        while ita < time_ita:
            if ita < 200:
                v_plt[ita] = self.c
            else:
                v_plt[ita] = v
            u_plt[ita] = u
            # dt = tstep
            # dU/dt = a * (b * (V - v_rest) - u)
            # dV/dt = (K/C)*(V-v_rest)*(V-v_threshold) - U/C + current[ita]/C
            v += tstep * ((self.K/self.C)*(v-self.v_rest)*(v-self.v_threshold)-u/self.C + current[ita]/self.C)
            u += tstep * self.a * (self.b * (v - self.v_rest) - u)
            if v > self.v_peak:
                if ita > 200:
                    spike[ita] = 1
                    num_spikes += 1
                v = self.c
                u += self.d

                # spike_times.append(ita)

            ita += 1
        time = np.arange(time_ita) * tstep
        i = 0
        for t in time:
            if spike[i] == 1:
                spike_times.append(t)
            i += 1
        return time, v_plt, spike, num_spikes, spike_times

    def append_pre_synapse_times(self, times):
        self.pre_syn = times

    def append_post_synapse_times(self, times):
        self.post_syn = times

    def append_synapse(self, synapse):
        self.synapses.append(synapse)
