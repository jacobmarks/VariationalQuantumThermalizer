import numpy as np
import scipy.linalg as LA
import copy

from utils import *

####################################################

class LatentDistribution(object):
	def __init__(self, nqubits):
		'''Base class for all latent distributions'''
		self.nqubits = nqubits

	def evaluate_entropy_at(self, params):
		pass

	def differentiate_entropy_at(self, params, diff_ind):
		pass

class NumericalLatentDistribution(object):
	def __init__(self, nqubits):
		'''Base subclass for all numerically defined
		latent distributions'''
		self.nqubits = nqubits

	def evaluate_entropy_at(self, params):
		dd = self.generate_diag_dist_at(params)
		return -np.real(np.dot(dd, np.log(dd)))

	def differentiate_entropy_at(self, params, diff_ind):
		dlt = 1E-9 ### uses finite-difference gradient
		left_params, right_params = params.copy(), params.copy()
		left_params[diff_ind] -= dlt/2.
		right_params[diff_ind] += dlt/2.
		left_S = self.evaluate_entropy_at(left_params)
		right_S = self.evaluate_entropy_at(right_params)
		return (right_S - left_S)/dlt

class FullLatentDistribution(NumericalLatentDistribution):
	def __init__(self, nqubits):
		'''Latent distribution in accordance with Eq. (12) in 
		https://arxiv.org/pdf/1910.02071.pdf''' 
		self.nqubits = nqubits
		self.nparams = 2 ** nqubits - 1

	def generate_diag_dist_at(self, params):
		return partition_normalize(np.ndarray.flatten(np.append(params, [0.5])))

class FactorizedLatentDistribution(NumericalLatentDistribution):
	def __init__(self, nqubits):
		'''Latent distribution in accordance with Eq. (5) in 
		https://arxiv.org/pdf/1910.02071.pdf'''
		self.nqubits = nqubits
		self.nparams = nqubits

	def generate_diag_dist_at(self, params):
		tmp = np.full(self.nparams, 0.5)
		thetas = np.reshape(np.ravel(np.column_stack((params,tmp))), (self.nparams, 2))
		dd = thetas[0]
		for i in range(1, self.nqubits):
			dd = np.kron(dd, thetas[i])
		return partition_normalize(dd)


# ###########################################################################
# ###########################  *** NOTE ***   ###############################
# #### This is where a classical NN or Energy-Based Model could be used ####
# #### Just define another subclass of LatentDistribution, 
# #### AnalyticLatentDistribution, with a different 
# #### differentiate_entropy_at() method.This also entails overriding the
# #### get_lat_obs_exp_gradient_at() method of Circuit.
# ###########################  *** NOTE ***   ###############################
# ###########################################################################


