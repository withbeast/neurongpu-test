import sys
sys.path.append('../../pythonlib')

import math
import argparse
import neurongpu as ngpu


def make_brunel(N):
	NP = N // 10 * 5
	NE = N // 10 * 4
	NI = N // 10 * 1
	N = NP + NE + NI

	Wex = 0.1 * 20000 / N
	Win = -0.5 * 20000 / N

	pg = ngpu.Create("poisson_generator")
	ngpu.SetStatus(pg, "rate", 20)

	neuron = ngpu.Create("iaf_psc_exp", N, 2)
	P = neuron[0:NP]      # excitatory neurons
	E = neuron[NP:NP+NE]   # inhibitory neurons
	I = neuron[NP+NE:N]

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


def make_vogels(N):
	NE = N // 5 * 4
	NI = N // 5 * 1
	N = NE + NI

	Wex = 0.4 * 16000000 / N / N
	Win = 5.1 * 16000000 / N / N

	neuron = ngpu.Create("iaf_psc_exp_g", N, 2)
	E = neuron[0:NE]   # inhibitory neurons
	I = neuron[NE:N]

	ngpu.SetStatus(neuron, {"V_m_rel": -60, "V_reset_rel": -60, "Theta_rel": -50, "t_ref": 5, "G_ex": 0, "G_in": 0})

	ngpu.Connect(E, E,
		{"rule": "fixed_total_number", "total_num": NE * NE // 50},
		{"weight": Wex, "delay": 0.8, "receptor":0})

	ngpu.Connect(E, I,
		{"rule": "fixed_total_number", "total_num": NE * NI // 50},
		{"weight": Wex, "delay": 0.8, "receptor":0})

	ngpu.Connect(I, E,
		{"rule": "fixed_total_number", "total_num": NI * NE // 50},
		{"weight": Win, "delay": 0.8, "receptor":1})

	ngpu.Connect(I, I,
		{"rule": "fixed_total_number", "total_num": NI * NI // 50},
		{"weight": Win, "delay": 0.8, "receptor":1})


def make_synth(N, pconnect, pfire, delay):
	pg = ngpu.Create("poisson_generator")
	ngpu.SetStatus(pg, "rate", 10000 * pfire)

	neuron = ngpu.Create("iaf_psc_exp", N, 2)

	ngpu.SetStatus(neuron, {"V_m_rel": 0, "V_reset_rel": 0, "Theta_rel": 20, "t_ref": 0})

	ngpu.Connect(pg, neuron, {"rule": "all_to_all"}, {"weight": 20, "delay": 0.1, "receptor":0})

	ngpu.Connect(neuron, neuron,
		{"rule": "fixed_total_number", "total_num": int(N * N * pconnect)},
		{"weight": 0, "delay": 0.1 * delay, "receptor":0})


params = argparse.ArgumentParser()
params.add_argument("--model", type=str)
params.add_argument("--nsyn", type=int)
params.add_argument("--pconnect", type=float)
params.add_argument("--pfire", type=float)
params.add_argument("--delay", type=int)
params, trash = params.parse_known_args()

ngpu.SetRandomSeed(1234) # seed for GPU random numbers

model = ""
if params.model == "vogels":
	model = "vogels"
	make_vogels(int(math.sqrt(params.nsyn / 0.02)))
elif params.model == "brunel":
	make_brunel(int(math.sqrt(params.nsyn / 0.05)))
	model = "brunel"
elif params.model == "synth":
	make_synth(int(math.sqrt(params.nsyn / params.pconnect)), params.pconnect, params.pfire, params.delay)
	model = "synth_" + str(params.pconnect) + "_" + str(params.pfire) + "_" + str(params.delay)

print("{")
print("\t\"sim\": \"NeuronGPU\",")
print("\t\"model\": \"" + model + "\",")
print("\t\"#syn\": " + str(params.nsyn) + ",")
print("\t\"#gpus\": 1,")

ngpu.Simulate()

print("}")