from random import shuffle
import os
import sys
import Utils
import Network
import pyspike as spk
import sys
sys.path.append('/home/aiswarya/PySpike/pyspike')
from pyspike import SpikeTrain
from datetime import datetime
from matplotlib import pyplot as plt
from neuronpy.graphics import spikeplot


prototype_trains = [None] * 10

def write_weights(network):
    i = 0
    with open("weights.txt", "a") as f:
        for out in network.output_layer:
            if i == 0:
                f.write("0\n")
            elif i == 1:
                f.write("1\n")
            elif i == 2:
                f.write("2\n")
            elif i == 3:
                f.write("3\n")
            elif i == 4:
                f.write("4\n")
            elif i == 5:
                f.write("5\n")
            elif i == 6:
                f.write("6\n")
            elif i == 7:
                f.write("7\n")
            elif i == 8:
                f.write("8\n")
            elif i == 9:
                f.write("9\n")
            
            for syn in out.synapses:
                if i == 0:
                    f.write("%s\n" % syn.w)
                    i = 1
                else:
                    f.write("%s\n" % syn.w)
                    i = 0
            i += 1

def print_result(results):
    print('\t0: ' + str(results[0]))
    print('\t1: ' + str(results[1]))
    print('\t2: ' + str(results[2]))
    print('\t3: ' + str(results[3]))
    print('\t4: ' + str(results[4]))
    print('\t5: ' + str(results[5]))
    print('\t6: ' + str(results[6]))
    print('\t7: ' + str(results[7]))
    print('\t8: ' + str(results[8]))
    print('\t9: ' + str(results[9]))
    

# Generate a spike train from the given spike
def generate_prototypes(spike, key):
    global prototype_trains
    spike_train = SpikeTrain(spike, [0.0, 300.0])
    if key == '0':
        prototype_trains[0] = spike_train
    elif key == '1':
        prototype_trains[1] = spike_train
    elif key == '2':
        prototype_trains[2] = spike_train
    elif key == '3':
        prototype_trains[3] = spike_train
    elif key == '4':
        prototype_trains[4] = spike_train
    elif key == '5':
        prototype_trains[5] = spike_train
    elif key == '6':
        prototype_trains[6] = spike_train
    elif key == '7':
        prototype_trains[7] = spike_train
    elif key == '8':
        prototype_trains[8] = spike_train
    elif key == '9':
        prototype_trains[9] = spike_train

def spike_analysis(spikes, value):
    distances = []
    i = 0
    for spike in spikes:
        spike_train = SpikeTrain(spike, [0.0, 300.0])
        isi_profile = spk.spike_sync(prototype_trains[i], spike_train)
        distances.append(isi_profile)
        i += 1

    val, idx = max((val, idx) for (idx, val) in enumerate(distances))
    print("Distance: %.8f" % val)
    print("Index: %s" % idx)


def show_plots(time, v_plts, currents, spikes):
    plt.figure('0')
    plt.plot(time, v_plts[0], 'g-')
    plt.plot(time, currents[0], 'r-')
    plt.figure('1')
    plt.plot(time, v_plts[1], 'b-')
    plt.plot(time, currents[1], 'y-')
    plt.figure('2')
    plt.plot(time, v_plts[2], 'k-')
    plt.plot(time, currents[2], 'm-')
    plt.figure('All')
    plt.plot(time, v_plts[0], 'g-')
    plt.plot(time, currents[0], 'r-')
    plt.plot(time, v_plts[1], 'b-')
    plt.plot(time, currents[1], 'y-')
    plt.plot(time, v_plts[2], 'k-')
    plt.plot(time, currents[2], 'm-')
    sp = spikeplot.SpikePlot()
    sp.plot_spikes(spikes)
    plt.show()

# Test our network
def test():
    print('Testing started')
    global prototype_trains
    prototype_trains = [None] * 10
    mapping = dict()

    weights = "weights.txt"

    network = Network.Network(weights=weights)
    audio_path = "/home/aiswarya/SNN_works/my_code/spoken-digit-dataset/free-spoken-digit-dataset-master/test"

    audio = [os.path.join(root, name)
                 for root, dirs, files in os.walk(audio_path)
                 for name in files
                 if name.endswith((".wav"))]

    for fname in audio:
        mapping[fname] = Utils.get_label(fname)

        _0_count = 1
        _1_count = 1
        _2_count = 1
        _3_count = 1
        _4_count = 1
        _5_count = 1
        _6_count = 1
        _7_count = 1
        _8_count = 1
        _9_count = 1

        count = 10
        for key in mapping:
            if mapping[key] == '0' and _0_count != 0:
                print(key)
                results, currents, time, v_plts, spikes = network.start(key)
                print_result(results)
                _0_count -= 1
                count -= 1
                if _0_count == 0:
                    # Generate a spike train for the '0' sound
                    generate_prototypes(spikes[0], '0')
                elif _0_count == 0 and _1_count == 0 and _2_count == 0 and _3_count == 0 and _4_count == 0 and _5_count == 0 and _6_count == 0 and _7_count == 0 and _8_count == 0 and _9_count == 0:
                    spike_analysis(spikes)
                    #show_plots(time, v_plts, currents, spikes)
                    #spike_analysis(spikes)
            elif mapping[key] == '1' and _1_count != 0:
                print(key)
                results, currents, time, v_plts, spikes = network.start(key)
                print_result(results)
                _1_count -= 1
                count -= 1
                if _1_count == 0:
                    # Generate a spike train for the '1' sound
                    generate_prototypes(spikes[1], '1')
                elif _0_count == 0 and _1_count == 0 and _2_count == 0 and _3_count == 0 and _4_count == 0 and _5_count == 0 and _6_count == 0 and _7_count == 0 and _8_count == 0 and _9_count == 0:
                    spike_analysis(spikes)
                    #show_plots(time, v_plts, currents, spikes)
                    #spike_analysis(spikes)
            elif mapping[key] == '2' and _2_count != 0:
                print(key)
                results, currents, time, v_plts, spikes = network.start(key)
                print_result(results)
                _2_count -= 1
                count -= 1
                if _2_count == 0:
                    # Generate a spike train for the '2' sound
                    generate_prototypes(spikes[2], '2')
                elif _0_count == 0 and _1_count == 0 and _2_count == 0 and _3_count == 0 and _4_count == 0 and _5_count == 0 and _6_count == 0 and _7_count == 0 and _8_count == 0 and _9_count == 0:
                    spike_analysis(spikes)
            elif mapping[key] == '3' and _3_count != 0:
                print(key)
                results, currents, time, v_plts, spikes = network.start(key)
                print_result(results)
                _3_count -= 1
                count -= 1
                if _3_count == 0:
                    # Generate a spike train for the '3' sound
                    generate_prototypes(spikes[3], '3')
                elif _0_count == 0 and _1_count == 0 and _2_count == 0 and _3_count == 0 and _4_count == 0 and _5_count == 0 and _6_count == 0 and _7_count == 0 and _8_count == 0 and _9_count == 0:
                    spike_analysis(spikes)
                    #show_plots(time, v_plts, currents, spikes)
                    #spike_analysis(spikes)
            elif mapping[key] == '4' and _4_count != 0:
                print(key)
                results, currents, time, v_plts, spikes = network.start(key)
                print_result(results)
                _4_count -= 1
                count -= 1
                if _4_count == 0:
                    # Generate a spike train for the '4' sound
                    generate_prototypes(spikes[4], '4')
                elif _0_count == 0 and _1_count == 0 and _2_count == 0 and _3_count == 0 and _4_count == 0 and _5_count == 0 and _6_count == 0 and _7_count == 0 and _8_count == 0 and _9_count == 0:
                    spike_analysis(spikes)
            elif mapping[key] == '5' and _5_count != 0:
                print(key)
                results, currents, time, v_plts, spikes = network.start(key)
                print_result(results)
                _5_count -= 1
                count -= 1
                if _5_count == 0:
                    # Generate a spike train for the '5' sound
                    generate_prototypes(spikes[5], '5')
                elif _0_count == 0 and _1_count == 0 and _2_count == 0 and _3_count == 0 and _4_count == 0 and _5_count == 0 and _6_count == 0 and _7_count == 0 and _8_count == 0 and _9_count == 0:
                    spike_analysis(spikes)
                    #show_plots(time, v_plts, currents, spikes)
                    #spike_analysis(spikes)
            elif mapping[key] == '6' and _6_count != 0:
                print(key)
                results, currents, time, v_plts, spikes = network.start(key)
                print_result(results)
                _6_count -= 1
                count -= 1
                if _6_count == 0:
                    # Generate a spike train for the '6' sound
                    generate_prototypes(spikes[6], '6')
                elif _0_count == 0 and _1_count == 0 and _2_count == 0 and _3_count == 0 and _4_count == 0 and _5_count == 0 and _6_count == 0 and _7_count == 0 and _8_count == 0 and _9_count == 0:
                    spike_analysis(spikes)
            elif mapping[key] == '7' and _7_count != 0:
                print(key)
                results, currents, time, v_plts, spikes = network.start(key)
                print_result(results)
                _7_count -= 1
                count -= 1
                if _7_count == 0:
                    # Generate a spike train for the '7' sound
                    generate_prototypes(spikes[7], '7')
                elif _0_count == 0 and _1_count == 0 and _2_count == 0 and _3_count == 0 and _4_count == 0 and _5_count == 0 and _6_count == 0 and _7_count == 0 and _8_count == 0 and _9_count == 0:
                    spike_analysis(spikes)
                    #show_plots(time, v_plts, currents, spikes)
                    #spike_analysis(spikes)
            elif mapping[key] == '8' and _8_count != 0:
                print(key)
                results, currents, time, v_plts, spikes = network.start(key)
                print_result(results)
                _8_count -= 1
                count -= 1
                if _8_count == 0:
                    # Generate a spike train for the '8' sound
                    generate_prototypes(spikes[8], '8')
                elif _0_count == 0 and _1_count == 0 and _2_count == 0 and _3_count == 0 and _4_count == 0 and _5_count == 0 and _6_count == 0 and _7_count == 0 and _8_count == 0 and _9_count == 0:
                    spike_analysis(spikes)
            elif mapping[key] == '9' and _9_count != 0:
                print(key)
                results, currents, time, v_plts, spikes = network.start(key)
                print_result(results)
                _9_count -= 1
                count -= 1
                if _9_count == 0:
                    # Generate a spike train for the '9' sound
                    generate_prototypes(spikes[9], '9')
                elif _0_count == 0 and _1_count == 0 and _2_count == 0 and _3_count == 0 and _4_count == 0 and _5_count == 0 and _6_count == 0 and _7_count == 0 and _8_count == 0 and _9_count == 0:
                    spike_analysis(spikes)
                    #show_plots(time, v_plts, currents, spikes)
                    #spike_analysis(spikes)
            
            elif count == 0:
                print(key)
                value = ''
                if mapping[key] == '0':
                    value = '0'
                elif mapping[key] == '1':
                    value = '1'
                elif mapping[key] == '2':
                    value = '2'
                elif mapping[key] == '3':
                    value = '3'
                elif mapping[key] == '4':
                    value = '4'
                elif mapping[key] == '5':
                    value = '5'
                elif mapping[key] == '6':
                    value = '6'
                elif mapping[key] == '7':
                    value = '7'
                elif mapping[key] == '8':
                    value = '8'
                elif mapping[key] == '9':
                    value = '9'

                results, currents, time, v_plts, spikes = network.start(key)
                print_result(results)
                spike_analysis(spikes, value)


# Train the network
def train():
    print('training started')
    network = Network.Network(weights=None)

    mapping = dict()

    audio_path = "/home/aiswarya/SNN_works/my_code/spoken-digit-dataset/free-spoken-digit-dataset-master/recordings"

    # Gets list of all audio files in the directory
    audio = [os.path.join(root, name)
             for root, dirs, files in os.walk(audio_path)
             for name in files
             if name.endswith((".wav"))]

    shuffle(audio)

    # Get a mapping of labels to audio
    for fname in audio:
        mapping[fname] = Utils.get_label(fname)

    print(datetime.now())

    _0_count = 150
    _1_count = 150
    _2_count = 150
    _3_count = 150
    _4_count = 150
    _5_count = 150
    _6_count = 150
    _7_count = 150
    _8_count = 150
    _9_count = 150
    

    i = 0
    for key in mapping:
        if(i%10 == 0):
            print(i)
        if mapping[key] == '0' and _0_count > 0:
            print(key)
            results, currents, time, v_plts, spikes = network.start(key)
            print_result(results)
            network.conduct_training(0)
            _0_count -= 1
        elif mapping[key] == '1' and _1_count > 0:
            print(key)
            results, currents, time, v_plts, spikes = network.start(key)
            print_result(results)
            network.conduct_training(1)
            _1_count -= 1
        elif mapping[key] == '2' and _2_count > 0:
            print(key)
            results, currents, time, v_plts, spikes = network.start(key)
            print_result(results)
            network.conduct_training(2)
            _2_count -= 1
        elif mapping[key] == '3' and _3_count > 0:
            print(key)
            results, currents, time, v_plts, spikes = network.start(key)
            print_result(results)
            network.conduct_training(3)
            _3_count -= 1
        elif mapping[key] == '4' and _4_count > 0:
            print(key)
            results, currents, time, v_plts, spikes = network.start(key)
            print_result(results)
            network.conduct_training(4)
            _4_count -= 1
        elif mapping[key] == '5' and _5_count > 0:
            print(key)
            results, currents, time, v_plts, spikes = network.start(key)
            print_result(results)
            network.conduct_training(5)
            _5_count -= 1
        elif mapping[key] == '6' and _6_count > 0:
            print(key)
            results, currents, time, v_plts, spikes = network.start(key)
            print_result(results)
            network.conduct_training(6)
            _6_count -= 1
        elif mapping[key] == '7' and _7_count > 0:
            print(key)
            results, currents, time, v_plts, spikes = network.start(key)
            print_result(results)
            network.conduct_training(7)
            _7_count -= 1
        elif mapping[key] == '8' and _8_count > 0:
            print(key)
            results, currents, time, v_plts, spikes = network.start(key)
            print_result(results)
            network.conduct_training(8)
            _8_count -= 1
        elif mapping[key] == '9' and _9_count > 0:
            print(key)
            results, currents, time, v_plts, spikes = network.start(key)
            print_result(results)
            network.conduct_training(9)
            _9_count -= 1

        i = i+1

    write_weights(network)
    print(datetime.now())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == 'train':
            print('Training...')
            train()
        else:
            print('Testing')
            test()


