import numpy as np
import scipy.linalg as LA
import matplotlib.pyplot as plt

from utils import *
from optimization_engine import *

####################################################

class VQTEngine(object):
	'''Engine for Variational Quantum Thermalizer Algorithm
		inputs: 
			- circuit structure
			- system hamiltonian H
			- inverse temp beta
			- variance in gaussian noise
			- nmeas = num measurements per expectation
			'''
	def __init__(self, circuit, H, beta, noise = None, nmeas = 10):
		self.circuit = circuit
		self.H = H
		self.beta = beta
		self.rho_thermal = self.get_thermal_state(H, beta)
		self.nparams = circuit.nparams
		self.noise = noise
		if noise == None:
			self.nmeas = 1
		else:
			self.nmeas = nmeas

	def initialize_params(self):
		self.set_params(3 * np.random.random(self.nparams) - 1.)

	def set_params(self, params):
		self.params = params

	def compute_loss(self, params):
		S = self.circuit.evaluate_entropy_at(params)
		E = self.circuit.evaluate_obs_expectation_at(params, self.H)
		return -S + self.beta * E

	def compute_gradient(self, params):
		dS = self.circuit.get_entropy_gradient_at(params)
		dE = self.circuit.get_obs_exp_gradient_at(params, self.H, noise = self.noise, nmeas = self.nmeas)
		return -dS + self.beta * dE

	def get_thermal_state(self, H, beta):
		rho = LA.expm(- beta * H)
		rho/= np.trace(rho)
		return rho

	def get_circuit_rho(self):
		return self.circuit.get_rho_at(self.params, noise = None)

	def run(self, optimizer = None, save_hist = True, verbose = True):
		if optimizer == None:
			optimizer = AdamOptimizer()
		self.initialize_params()
		if save_hist == True:
			self.loss_history = []
		else:
			self.loss_history = False


		while optimizer.t < optimizer.maxiter:
			if save_hist == True:
				self.loss_history += [self.compute_loss(self.params)]
			if optimizer.t % 50 == 0 and verbose == True:
				print("t = {} | loss = {}".format(optimizer.t, self.loss_history[-1]))
			grad = self.compute_gradient(self.params)
			update = optimizer.compute_update(grad)
			if np.mean(np.abs(update)) < 3E-6:
				break
			else:
				self.params += update
		print('Final loss: {}'.format(self.loss_history[-1]))

	def plot_loss(self, ax, include_target = True):
		if self.loss_history == False:
			print('Loss history was not saved')
			return

		ax.set_xlabel(r'Iteration $t$')
		if include_target:
			ax.plot(self.loss_history, label = r'$\mathcal{L}(t)$')
			rho_thermal = get_thermal_state(self.H, self.beta)
			loss_target = get_free_energy(rho_thermal, self.beta, self.H)
			ax.axhline(y = loss_target, ls = '--', label = 'Target loss')
			ax.legend()
		else:
			ax.plot(self.loss_history)
			ax.set_ylabel(r'$\mathcal{L}(t)$')






###########################################################################






