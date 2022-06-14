import numpy as np
import scipy.linalg as LA
import copy

from utils import *
from gateset import *
#######################################################

class GateLayer(object):
	def __init__(self, gates, nqubits):
		'''Assuming only single qubit and neighboring two qubit gates '''
		self.nqubits = nqubits
		self.gen_ordered_gates(gates, nqubits)

	def copy(self):
		return copy.deepcopy(self)

	def replace_gate(self, g, ind):
		'''replaces gate at position ind with g'''
		self.ordered_gates[ind] = g

	def get_qubit_gate(self, gates, q):
		'''return the gate acting on qubit q'''
		for g in gates:
			if q in g.qubits:
				return g

	def get_gate_qubits(self, gates):
		'''return a list of all qubits acted on
		by nontrivial gates in this gate layer.'''
		qs = []
		for g in gates:
			qs += g.qubits
		return qs

	def gen_ordered_gates(self, gates, nqubits):
		ordered_gates = []
		ordered_nparams = []
		qubit_gate_dic = {}
		param_gate_dic = {}
		gate_qubits = self.get_gate_qubits(gates)
		if len(sorted(set(gate_qubits))) < len(gate_qubits):
			print('Improper Layer! A qubit is used in multiple gates')
			exit(0)
		gate_qubits = list(set(gate_qubits))
		q = 0
		while q < nqubits:
			if q not in gate_qubits:
				ordered_gates += [IdentityGate([q])]
				ordered_nparams += [0]
				qubit_gate_dic[q] = len(ordered_gates) - 1
				q += 1
			else:
				g = self.get_qubit_gate(gates, q)
				ordered_gates += [g]
				ordered_nparams += [g.ninput]
				if len(g.qubits) == 1:
					qubit_gate_dic[q] = len(ordered_gates) - 1
					q += 1
				else:
					qubit_gate_dic[q] = len(ordered_gates) - 1
					qubit_gate_dic[q + 1] = len(ordered_gates) - 1
					q += 2
		params_counter = 0
		for i in range(len(ordered_gates)):
			for j in range(ordered_nparams[i]):
				param_gate_dic[params_counter + j] = i
			params_counter += ordered_nparams[i]

		self.ordered_gates = ordered_gates
		self.ordered_nparams = ordered_nparams
		self.qubit_gate_dic = qubit_gate_dic
		self.param_gate_dic = param_gate_dic
		self.nparams = np.sum(ordered_nparams)
		self.gate_start_params = np.cumsum(np.array([0] + self.ordered_nparams))[:-1]
		self.gate_stop_params = np.cumsum(np.array([0] + self.ordered_nparams))[1:]

		

	def evaluate_at(self, params, noise = None):
		if len(self.ordered_gates) == 1:
			return self.ordered_gates[0].evaluate_at(params,  noise = noise)
		else:
			par_ind = 0
			npar = self.ordered_nparams[0]
			if npar == 0:
				u = self.ordered_gates[0].evaluate_at([],  noise = noise)
			else:
				u = self.ordered_gates[0].evaluate_at(params[:npar],  noise = noise)
				par_ind += npar


			for i in range(1, len(self.ordered_gates)):
				npar = self.ordered_nparams[i]
				g = self.ordered_gates[i]
				u = np.kron(u, g.evaluate_at(params[par_ind: par_ind + npar], noise = noise))
				par_ind += npar
			return u


	def get_differential_layers(self, diff_param_ind):
		diff_gate_ind = self.param_gate_dic[diff_param_ind]
		diff_gate_start = self.gate_start_params[diff_gate_ind]
		diff_gate_stop = self.gate_stop_params[diff_gate_ind]
		diff_gate_param_ind = diff_param_ind - diff_gate_start
		g = self.ordered_gates[diff_gate_ind]

		diff_gates, weights = g.get_differential_gates(diff_gate_param_ind)
		diff_layers = [self.copy() for j in range(len(diff_gates))]
		for i, dg in enumerate(diff_gates):
			diff_layers[i].replace_gate(dg, diff_gate_ind)
		return diff_layers, weights




# ############################################################################

def SingleQubitLayer(nqubits):
	gates = [SingleQubitRotationGate([q]) for q in range(nqubits)]
	return GateLayer(gates, nqubits)

def TwoQubitLayer(nqubits, offset = False):
	def get_qubit_pairs(nqubits, offset):
		pairs = []
		if offset == True:
			q = 1
		else:
			q = 0
		while q+1 < nqubits:
			pairs += [[q, q+1]]
			q += 2
		return pairs

	pairs = get_qubit_pairs(nqubits, offset)
	gates = [IsingCouplingGate(p) for p in pairs]
	return GateLayer(gates, nqubits)

# ###########################################################################

class CircuitUnitary(object):
	def __init__(self, layers):
		self.nqubits = layers[0].nqubits
		self.layers = layers
		self.nlayers = len(layers)
		self.gen_layer_param_mappings()

	def copy(self):
		return copy.deepcopy(self)

	def replace_layer(self, l, ind):
		'''replaces layer at position ind with l'''
		self.layers[ind] = l

	def gen_layer_param_mappings(self):
		# identify which layer a parameter is in
		# and the param num in a layer
		layer_num_dic = {}
		param_num_in_layer_dic = {}
		layer_start_params = []
		layer_stop_params = []
		param_counter = 0
		for i, layer in enumerate(self.layers):
			layer_start_params += [param_counter]
			nparams = layer.nparams
			for j in range(nparams):
				layer_num_dic[param_counter] = i
				param_num_in_layer_dic[param_counter] = j
				param_counter += 1
			layer_stop_params += [param_counter]
		self.layer_num_dic = layer_num_dic
		self.param_num_in_layer_dic = param_num_in_layer_dic
		self.layer_start_params = layer_start_params
		self.layer_stop_params = layer_stop_params
		self.nparams = self.layer_stop_params[-1]

	def get_layer_unitaries(self, params, noise):
		us = []
		for i, layer in enumerate(self.layers):
			start, stop = self.layer_start_params[i], self.layer_stop_params[i]
			us += [layer.evaluate_at(params[start:stop], noise)]
		return us

	def evaluate_at(self, params, noise = None):
		us = self.get_layer_unitaries(params, noise)
		u = us[0]
		for i in range(1, len(us)):
			u = np.matmul(u, us[i])
		return u

	def get_differential_unitaries(self, diff_param_ind):
		p_in_lay_ind = self.param_num_in_layer_dic[diff_param_ind]
		lay_num = self.layer_num_dic[diff_param_ind]
		l = self.layers[lay_num]

		diff_layers, weights = l.get_differential_layers(p_in_lay_ind)
		diff_unitaries = [self.copy() for i in range(len(diff_layers))]
		for i, dl in enumerate(diff_layers):

			diff_unitaries[i].replace_layer(dl, lay_num)
		return diff_unitaries, weights

############################################################################


