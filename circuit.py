import numpy as np
import copy

from utils import *
from latent import *
from unitary import *
# ###########################################################################

class Circuit(object):
	'''circuit object containing structure for the
		unitary operation and the classical latent distribution'''
	def __init__(self, circ_unitary, latent_dist):
		self.nqubits = latent_dist.nqubits
		self.nlatent_params = latent_dist.nparams
		self.nunitary_params = circ_unitary.nparams
		self.nparams = self.nlatent_params + self.nunitary_params
		self.circ_unitary = circ_unitary
		self.latent_dist = latent_dist

	def evaluate_entropy_at(self, params):
		lat_params = params[:self.nlatent_params]
		s = self.latent_dist.evaluate_entropy_at(lat_params)
		return s

	def get_entropy_gradient_at(self, params):
		lat_params = params[:self.nlatent_params]
		s_grads = np.zeros_like(params)
		for diff_ind in range(len(lat_params)):
			s_grads[diff_ind] = self.latent_dist.differentiate_entropy_at(lat_params, diff_ind)
		return s_grads

	def get_rho_at(self, params, noise):
		lat_params, uni_params = params[:self.nlatent_params], params[self.nlatent_params:]
		rho = np.diag(self.latent_dist.generate_diag_dist_at(lat_params))
		u = self.circ_unitary.evaluate_at(uni_params, noise)
		rho = apply_unitary_to_state(rho, u)
		return rho

	def evaluate_obs_expectation_at(self, params, obs, noise = None, nmeas = 1):
		'''Compute the expectation value of an observable obs
			evaluated at the input params'''
		if noise == None:
			nmeas = 1
		exps = np.zeros(nmeas)
		for i in range(nmeas):
			rho = self.get_rho_at(params, noise)
			exps[i] = get_expectation(rho, obs)
		return np.mean(exps)

	def get_lat_obs_exp_gradient_at(self, lat_params, uni_params, obs, noise, nmeas):
		if noise == None:
			nmeas = 1
		lat_grads = np.zeros_like(lat_params)
		dlt = 1E-9
		for diff_ind in range(self.nlatent_params):
			left_params, right_params = lat_params.copy(), lat_params.copy()
			left_params[diff_ind] -= dlt/2.
			right_params[diff_ind] += dlt/2.
			left_rho = np.diag(self.latent_dist.generate_diag_dist_at(left_params))
			right_rho = np.diag(self.latent_dist.generate_diag_dist_at(right_params))

			exps = np.zeros(nmeas)
			for i in range(nmeas):
				u = self.circ_unitary.evaluate_at(uni_params)
				new_left_rho = apply_unitary_to_state(left_rho, u)
				new_right_rho = apply_unitary_to_state(right_rho, u)
				left_exp = get_expectation(new_left_rho, obs)
				right_exp = get_expectation(new_right_rho, obs)
				exps[i] = (right_exp - left_exp)

			lat_grads[diff_ind] = np.mean(exps)/dlt
		return lat_grads

	def get_uni_obs_exp_gradient_at(self, lat_params, uni_params, obs, noise, nmeas):
		if noise == None:
			nmeas = 1
		nqubits = self.nqubits
		rho_dd = np.diag(self.latent_dist.generate_diag_dist_at(lat_params))
		nparams = len(uni_params)

		grads = np.zeros(nparams)
		for i in range(nparams):
			dus, weights = self.circ_unitary.get_differential_unitaries(i)
			exps = np.zeros(nmeas)
			for j in range(nmeas):
				tmp = 0.
				for k, du in enumerate(dus):
					u = du.evaluate_at(uni_params, noise)
					rho = apply_unitary_to_state(rho_dd, u)
					tmp += weights[k] * get_expectation(rho, obs)
				exps[j] = tmp
			grads[i] = np.mean(exps)
		return grads

	def get_obs_exp_gradient_at(self, params, obs, noise = None, nmeas = 1):
		'''Compute the differential of an observable obs
			evaluated at the input params'''
		if noise == None:
			nmeas = 1
		lat_params, uni_params = params[:self.nlatent_params], params[self.nlatent_params:]
		lat_grads = self.get_lat_obs_exp_gradient_at(lat_params, uni_params, obs, noise, nmeas)
		uni_grads = self.get_uni_obs_exp_gradient_at(lat_params, uni_params, obs, noise, nmeas)
		return np.concatenate((lat_grads, uni_grads))

# ###########################################################################

def generate_circuit(nqubits, nlayers, factorized = False):
	'''Generated circuit has nlayers of single qubit gates interweaved
		with nlayers of two qubit gates, for a total of 2*nlayers layers'''
	if nlayers % 2 != 0:
		print('Number of layers must be divisible by 2.')
		exit(0)
	layers = []
	for i in range(int(nlayers//2)):
		layers += [SingleQubitLayer(nqubits), TwoQubitLayer(nqubits, offset = (i%2))]
	cu = CircuitUnitary(layers)

	if factorized == True:
		ld = FactorizedLatentDistribution(nqubits)
	else:
		ld = FullLatentDistribution(nqubits)

	return Circuit(cu, ld)
