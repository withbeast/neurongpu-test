import sys

sys.path.append('../../pythonlib')

import ctypes
import neurongpu as ngpu
from random import randrange

if len(sys.argv) != 2:
    print ("Usage: python %s n_neurons" % sys.argv[0])
    quit()
    
order = int(sys.argv[1])//10

ngpu.SetRandomSeed(1234) # seed for GPU random numbers

n_receptors = 2

NE = 4 * order       # number of excitatory neurons
NI = 1 * order       # number of inhibitory neurons
n_neurons = NE + NI  # number of neurons in total

Wex = 0.1
Win = -0.5

# create poisson generator
pg = ngpu.Create("poisson_generator")
ngpu.SetStatus(pg, "rate", 20)

# Create n_neurons neurons with n_receptor receptor ports
neuron = ngpu.Create("iaf_psc_exp", n_neurons, n_receptors)
exc_neuron = neuron[0:NE]      # excitatory neurons
inh_neuron = neuron[NE:n_neurons]   # inhibitory neurons
  
# receptor parameters
ngpu.SetStatus(neuron, {"V_m_rel": 0, "V_reset_rel": 0, "Theta_rel": 20, "t_ref": 0})

exc_conn_dict={"rule": "fixed_total_number", "total_num": NE * n_neurons // 10}
exc_syn_dict={"weight": 0, "delay": 1.5, "receptor":0}
ngpu.Connect(exc_neuron, neuron, exc_conn_dict, exc_syn_dict)

inh_conn_dict={"rule": "fixed_total_number", "total_num": NI * n_neurons // 10}
inh_syn_dict={"weight": 0, "delay": 1.5, "receptor":1}
ngpu.Connect(inh_neuron, neuron, inh_conn_dict, inh_syn_dict)

pg_conn_dict={"rule": "all_to_all"}
pg_syn_dict={"weight": 10, "delay": 1.5, "receptor":0}
ngpu.Connect(pg, neuron, pg_conn_dict, pg_syn_dict)

ngpu.Simulate()