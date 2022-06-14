import numpy as np
from scipy.linalg import eigh
import scipy.linalg as LA
import pickle
import scipy.optimize as optimize
import matplotlib.pyplot as plt

####################################################

# Define Pauli Ops
sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]])
sigma_y = np.array([[0.0, -1j], [1j, 0.0]])
sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]])
ident = np.eye(2)

def pauli_op(pauli_type):
	assert(pauli_type in ['X', 'Y', 'Z', 'I'])
	if pauli_type == 'X':
		return sigma_x
	elif pauli_type == 'Y':
		return sigma_y
	elif pauli_type == 'Z':
		return sigma_z
	else:
		return ident

####################################################


def gen_single_qubit_pauli(n_qubits, q, pauli_type):
	op = pauli_op(pauli_type)
	ops = [op if i == q else np.eye(2) for i in range(n_qubits)]
	if n_qubits == 1:
		return ops[0]
	many_body_op = np.kron(ops[0], ops[1])
	for i in range(2, n_qubits):
		many_body_op = np.kron(many_body_op, ops[i])
	return many_body_op

def gen_single_qubit_pauliX(n_qubits, q):
	return gen_single_qubit_pauli(n_qubits, q, 'X')

def gen_single_qubit_pauliY(n_qubits, q):
	return gen_single_qubit_pauli(n_qubits, q, 'Y')

def gen_single_qubit_pauliZ(n_qubits, q):
	return gen_single_qubit_pauli(n_qubits, q, 'Z')

####################################################

def gen_coupling(n_qubits, q1, q2,pauli_type):
	op = pauli_op(pauli_type)
	dim = 2 ** n_qubits
	ops = [op if i in [q1, q2] else np.eye(2) for i in range(n_qubits)]
	if n_qubits == 1:
		return ops[0]
	many_body_op = np.kron(ops[0], ops[1])
	for i in range(2, n_qubits):
		many_body_op = np.kron(many_body_op, ops[i])
	return many_body_op

def gen_xx_coupling(n_qubits, q1, q2):
	return gen_coupling(n_qubits, q1, q2, 'X')

def gen_yy_coupling(n_qubits, q1, q2):
	return gen_coupling(n_qubits, q1, q2, 'Y')

def gen_zz_coupling(n_qubits, q1, q2):
	return gen_coupling(n_qubits, q1, q2, 'Z')


####################################################

def gen_term_from_pauli_dict(n_qubits, pauli_dict):
	keys = list(pauli_dict.keys())
	if len(keys) == 1:
		q, p = keys[0], pauli_dict[keys[0]]
		return gen_single_qubit_pauli(n_qubits, q, p)
	elif len(keys) == 2:
		qs = keys
		p = pauli_dict[keys[0]]
		return gen_coupling(n_qubits, qs[0], qs[1], p)

def gen_ham_from_pauli_products(n_qubits, pauli_prods):
	dim = 2 ** n_qubits
	ham = np.zeros((dim, dim), dtype = np.complex_)
	for prod in pauli_prods:
		J, pauli_dict = prod[0], prod[1]
		ham += J * gen_term_from_pauli_dict(n_qubits, pauli_dict)

	if np.all(np.isreal(ham)):
		ham = np.real(ham)
	return ham

#### Heisenberg Hamiltonian ####
def gen_field_paulis(n_qubits, hx = False, hy = False, hz = False):
	pairs = []
	for i in range(n_qubits):
		if hx != False:    
			pairs += [(hx, {i: 'X'})]
		if hy != False:
			pairs += [(hy, {i: 'Y'})]
		if hz != False:
			pairs += [(hz, {i: 'Z'})]
	return pairs

#####################################################

def gen_heis1d_pauli_products(n_qubits, J = -1., hx = False, hy = False, hz = False, PBC_FLAG = False):
	pairs = []
	for i in range(n_qubits - 1):
		pairs += [(J, {i:'X', i+1:'X'})]
		pairs += [(J, {i:'Y', i+1:'Y'})]
		pairs += [(J, {i:'Z', i+1:'Z'})]

	pairs += gen_field_paulis(n_qubits, hx, hy, hz)
	return pairs

def gen_heisenberg1d_hamiltonian(n_qubits, J = -1., hx = False, hy = False, hz = False, PBC_FLAG = False):
	'''generates a 1d Heisenberg hamiltonian
	H = J sum_<ij> S_i.S_j +  sum_j H.S_j'''
	heis_pairs = gen_heis1d_pauli_products(n_qubits, J, hx, hy, hz, PBC_FLAG)
	return gen_ham_from_pauli_products(n_qubits, heis_pairs)

####################################################
####################################################
