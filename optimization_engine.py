import numpy as np

############################################################################

class AdamOptimizer(object):
	'''Optimization Engine that takes in gradients and spits out updates'''
	def __init__(self, alpha = 0.03, b1 = 0.9, b2 = 0.999, eps = 1e-7, maxiter = 1000):
		self.alpha = alpha
		self.b1 = b1
		self.b2 = b2
		self.eps = eps

		self.m_t = 0.
		self.v_t = 0.
		self.t = 0
		self.maxiter = maxiter
		self.success = False

	def compute_update(self, grad):
		self.t += 1
		self.m_t = self.b1 * self.m_t + (1 - self.b1) * grad
		self.v_t = self.b2 * self.v_t + (1 - self.b2) * np.square(grad)
		m_cap = self.m_t/(1 - (self.b1**self.t))
		v_cap = self.v_t/(1 - (self.b2**self.t))
		update = - self.alpha * m_cap / (np.sqrt(v_cap + self.eps))
		return update



###########################################################################