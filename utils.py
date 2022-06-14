import numpy as np
import scipy.linalg as LA

####################################################

def get_fidelity(rho, sigma):
	tmp = LA.sqrtm(rho)
	tmp = np.matmul(tmp, np.matmul(sigma, tmp))
	tmp = np.square(np.real(np.trace(LA.sqrtm(tmp))))
	return tmp

####################################################

def get_trace_distance(rho, sigma):
	tmp = rho - sigma
	tmp = np.matmul(np.conj(tmp).T, tmp)
	tmp = np.real(np.trace(LA.sqrtm(tmp)))/2.
	return tmp

####################################################

def get_free_energy(rho, beta, H):
	S = - np.trace(np.matmul(rho, LA.logm(rho)))
	E = get_expectation(rho, H)
	# print('E: {}'.format(np.round(E, 3)))
	# print('S: {}'.format(np.round(S, 3)))
	return np.real(-S + beta * E)


####################################################

def get_thermal_state(H, beta):
	rho = LA.expm(- beta * H)
	rho/= np.trace(rho)
	return rho

####################################################


def partition_normalize(arr):
	'''normalize as if arr contains energies in a 
		partition function'''
	tmp = np.exp(-arr)
	return tmp/np.sum(tmp)

####################################################

def gen_random_state(nqubits):
	dim = 2**nqubits
	psi = np.zeros(dim, dtype = np.complex_)
	psi += np.random.random(dim)
	psi += 1.j * np.random.random(dim)
	psi/= np.linalg.norm(psi)
	return psi

####################################################

def apply_unitary_to_state(rho, u):
	''' rho -> u rho u_dag '''
	return np.matmul(u, np.matmul(rho, np.conj(u).T))

####################################################

def get_expectation(rho, op):
	return np.real(np.trace(np.matmul(rho, op)))
	
####################################################


