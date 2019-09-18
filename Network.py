import numpy as np
import Synapse
import izhikevich_neuron
import Utils
from matplotlib import pyplot as plt
from neuronpy.graphics import spikeplot

global input_layer
global output_layer
global hidden_layer
global synapses
global a, b, c, d, time_ita

class Network:

    def __init__(self, weights):
        
        self.time_ita = 3000  # 100ms

        # Build a layer of 200 input neurons 
        self.input_layer = []
        for i in range(200):
            n = izhikevich_neuron.izhikevich()
            self.input_layer.append(n)

        # Build output layer, one neuron for each digit 
        self.output_layer = []
        for i in range(10):
            o = izhikevich_neuron.izhikevich()
            self.output_layer.append(o)

        i = 0
        # For each input neuron, append one synapse to each output neuron
        for n in self.input_layer:
            for out in self.output_layer:
                synapse = Synapse.Synapse()
                n.append_synapse(synapse)
                out.append_synapse(synapse)

        if weights is not None:
            i = 0
            n = 0
            s = 0
            with open(weights) as f:
                for line in f:
                    # For every 200 synapses, go to the next neuron
                    if i % 200 == 0 and n < len(self.output_layer):
                        s = 0
                        out = self.output_layer[n]
                        n += 1
                        out.synapses[s].set_weight(line)
                        s += 1
                    elif s < 200:
                        out.synapses[s].set_weight(line)
                        s += 1
                    i += 1

    # Calculates the current from the given MFCC value
    def get_current(self, x):
        return x

    # Equation 8 from the paper
    def get_total(self, neuron):
        g_tot = 0
        for synapse in neuron.synapses:
            t = 0
            i = 0
            tau = 2
            #print("Synapse")
            for j in synapse.spike:
                if j == 1:
                    t_kj = synapse.time[i]
                    
                    t = np.abs(t - t_kj)
                    #print("\t%s" % t)
                    g_tot += synapse.w * (t - t_kj) * np.exp(-(t - t_kj) / tau)
                    t = t_kj
                i += 1
        return g_tot


    # Get the total synaptic output for this neuron
    def total_synaptic_value(self, neuron):
        conductance = 0
        for syn_k in neuron.synapses:
            output = syn_k.synapse(2)
            conductance += output
        return conductance

    # if result == 0, then our target neuron is the first neuron in the output layer
    # result == 1 --> 2nd output neuron, result == 3 --> 3rd output neuron and so on
    def conduct_training(self, result):
        i = 0
        for out in self.output_layer:
            if i == result:
                # Undergo Hebbian STDP
                for syn in out.synapses:
                    syn.Heb_STDP()
            else:
                # Undergo anti-Hebbian STDP for non-target synapses
                for syn in out.synapses:
                    syn.Anti_Heb_STDP()
            i += 1


    # Perform analysis on the given filename using mel_Freq command
    def start(self, fname):

        features = Utils.get_features(fname)
        features = features[:200]

        #Feed features into our network and get spike information (number of spikes, time of largest spike)
        i = 0

        # Use for mel_freq. 520 input neurons
        for feature in features:
            n = self.input_layer[i]
            current = np.ones(self.time_ita) * self.get_current(feature)
            time, v_plt, spike, num_spikes, spike_times = n.neuron_simulation(self.time_ita, current)
            
            # Set pre spikes for each synapse connected to this neuron
            for synapse in n.synapses:
                synapse.set_pre_spikes(spike_times)
                synapse.set_time(time)
                synapse.set_spike(spike)
            i += 1


        # Create a 10 neuron output vector
        outputs = [0] * 10
        spikes = []
        v_plts = []
        currents = []
        i = 0
        for out in self.output_layer:
            current = np.ones(self.time_ita) * self.total_synaptic_value(out)
            time, v_plt, spike, num_spikes, spike_times = out.neuron_simulation(self.time_ita, current)
            spikes.append(spike_times)
            v_plts.append(v_plt)
            currents.append(current)
            for syn in out.synapses:
                syn.set_post_spikes(spike_times)

            outputs[i] = num_spikes
            i += 1

        return outputs, currents, time, v_plts, spikes