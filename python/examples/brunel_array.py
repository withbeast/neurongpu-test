import sys

sys.path.append('../../pythonlib')

import ctypes
import neurongpu as ngpu
from random import randrange

if len(sys.argv) != 2:
    print ("Usage: python %s n_neurons" % sys.argv[0])
    quit()
    
N = int(sys.argv[1])

ngpu.SetRandomSeed(1234) # seed for GPU random numbers

NP = N // 10 * 5
NE = N // 10 * 4
NI = N // 10 * 1
N = NP + NE + NI

Wex = 0.1 * 20000 / N
Win = -0.5 * 20000 / N

# create poisson generator
pg = ngpu.Create("poisson_generator")
ngpu.SetStatus(pg, "rate", 20)

# Create n_neurons neurons with n_receptor receptor ports
neuron = ngpu.Create("iaf_psc_exp", N, 2)
P = neuron[0:NP]      # excitatory neurons
E = neuron[NP:NP+NE]   # inhibitory neurons
I = neuron[NP+NE:N]

# receptor parameters
ngpu.SetStatus(P, {"V_m_rel": 0, "V_reset_rel": 0, "Theta_rel": 20, "t_ref": 0})
ngpu.SetStatus(E, {"V_m_rel": 0, "V_reset_rel": 0, "Theta_rel": 20, "t_ref": 2})
ngpu.SetStatus(I, {"V_m_rel": 0, "V_reset_rel": 0, "Theta_rel": 20, "t_ref": 2})

ngpu.Connect(pg, P, {"rule": "all_to_all"}, {"weight": 20, "delay": 0.1, "receptor":0})

ngpu.Connect(P, E,
	{"rule": "fixed_total_number", "total_num": NP * NE // 10},
	{"weight": Wex, "delay": 1.5, "receptor":0})

ngpu.Connect(P, I,
	{"rule": "fixed_total_number", "total_num": NP * NI // 10},
	{"weight": Wex, "delay": 1.5, "receptor":0})

ngpu.Connect(E, E,
	{"rule": "fixed_total_number", "total_num": NE * NE // 10},
	{"weight": Wex, "delay": 1.5, "receptor":0})

ngpu.Connect(E, I,
	{"rule": "fixed_total_number", "total_num": NE * NI // 10},
	{"weight": Wex, "delay": 1.5, "receptor":0})

ngpu.Connect(I, E,
	{"rule": "fixed_total_number", "total_num": NI * NE // 10},
	{"weight": Win, "delay": 1.5, "receptor":1})

ngpu.Connect(I, I,
	{"rule": "fixed_total_number", "total_num": NI * NI // 10},
	{"weight": Win, "delay": 1.5, "receptor":1})

ngpu.Simulate()