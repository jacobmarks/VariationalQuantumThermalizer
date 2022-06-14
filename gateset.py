import numpy as np
import copy

from utils import *
####################################################

XMat = np.array([[0.0, 1.0], [1.0, 0.0]])
YMat = np.array([[0.0, -1j], [1j, 0.0]])
ZMat = np.array([[1.0, 0.0], [0.0, -1.0]])
IMat = np.eye(2)
HMat =  np.array([[np.sqrt(0.5), np.sqrt(0.5)], [np.sqrt(0.5), -np.sqrt(0.5)]]) 

CNOTMat = np.array([
			[1., 0., 0., 0.],
			[0., 1., 0., 0.],
			[0., 0., 0., 1.],
			[0, 0, 1., 0.]])


def XPowMat(t):
	g = np.exp(1.j * np.pi * t /2.)
	c = np.cos(np.pi * t/2.)
	s = np.sin(np.pi * t/2.)
	return np.array([[g * c, - 1.j * g * s], [- 1.j * g * s, g * c]])

def YPowMat(t):
	g = np.exp(1.j * np.pi * t /2.)
	c = np.cos(np.pi * t/2.)
	s = np.sin(np.pi * t/2.)
	return np.array([[g * c, - g * s], [g * s, g * c]])

def ZPowMat(t):
	g = np.exp(-1.j * np.pi * t)
	return np.array([[1., 0.], [0., g]], dtype = np.complex_)

####################################################

def XXPowMat(t):
	Z = np.kron(IMat, ZPowMat(t))
	H1 = np.kron(HMat, IMat)
	H2 = np.kron(IMat, HMat)
	U = np.matmul(H1, H2)
	U = np.matmul(U, CNOTMat)
	U = np.matmul(U, Z)
	U = np.matmul(U, CNOTMat)
	U = np.matmul(U, H1)
	U = np.matmul(U, H2)
	return U

def YYPowMat(t):
	theta = np.pi * t
	return np.array([[np.cos(theta),0.,0.,-1j*np.sin(theta)],
			[0.,np.cos(theta),1j*np.sin(theta),0.],
			[0.,1j*np.sin(theta),np.cos(theta),0.],
			[-1j*np.sin(theta),0.,0.,np.cos(theta)]])

def ZZPowMat(t):
	Z = np.kron(IMat, ZPowMat(t))
	U = np.matmul(CNOTMat, Z)
	U = np.matmul(U, CNOTMat)
	return U

#############################################################################

class Gate(object):
	def __init__(self, qubits):
		self.qubits = qubits

	def copy(self):
		return copy.deepcopy(self)

	def initialize_offset(self):
		self.offset = np.zeros(self.nparams)

	def set_offset(self, offset):
		self.offset = offset

	def op_mat(self, ts):
		pass

	def add_offset(self, params):
		params += self.offset
		return params

	def add_noise(self, params, noise):
		if noise != None:
			params += np.random.normal(scale = noise, size = self.nparams)
		return params

	def format_params(self, ts):
		return np.array(ts.copy())

	def evaluate_at(self, ts, noise = None):
		params = self.format_params(ts)
		params = self.add_offset(params)
		params = self.add_noise(params, noise)
		return self.op_mat(params)

	def get_differential_gates(self, diff_ind):
		pass

#############################################################################

class IdentityGate(Gate):
	def __init__(self, qubits):
		self.qubits = qubits
		self.gate_string = 'I'
		self.nparams = 0
		self.ninput = 0
		self.initialize_offset()

	def op_mat(self, params):
		return IMat

	def get_differential_gates(self, diff_ind):
		gates, weights = [], []
		return gates, weights

#############################################################################

class RotAxisGate(Gate):
	def __init__(self, qubits):
		self.qubits = qubits
		self.nparams = 1
		self.ninput = 1
		self.initialize_offset()
		self.r = np.pi/2.

	def get_differential_gates(self, diff_ind):
		offset_left = np.array([- np.pi/(4 * self.r)])
		offset_right = np.array([np.pi/(4 * self.r)])

		weights = [self.r, -self.r]
		gleft = self.copy()
		gleft.set_offset(offset_left)
		gright = self.copy()
		gright.set_offset(offset_right)
		gates = [gright, gleft]
		return gates, weights

#############################################################################

class RxGate(RotAxisGate):
	def __init__(self, qubits):
		super().__init__(qubits)
		self.gate_string = 'Rx'

	def op_mat(self, params):
		return XPowMat(params[0])

class RyGate(RotAxisGate):
	def __init__(self, qubits):
		super().__init__(qubits)
		self.gate_string = 'Ry'

	def op_mat(self, params):
		return YPowMat(params[0])

class RzGate(RotAxisGate):
	def __init__(self, qubits):
		super().__init__(qubits)
		self.gate_string = 'Rz'

	def op_mat(self, params):
		return ZPowMat(params[0])

#############################################################################

class RxxGate(RotAxisGate):
	def __init__(self, qubits):
		super().__init__(qubits)
		self.gate_string = 'Rxx'

	def op_mat(self, params):
		return XXPowMat(params[0])

class RyyGate(RotAxisGate):
	def __init__(self, qubits):
		super().__init__(qubits)
		self.gate_string = 'Ryy'

	def op_mat(self, params):
		return YYPowMat(params[0])

class RzzGate(RotAxisGate):
	def __init__(self, qubits):
		super().__init__(qubits)
		self.gate_string = 'Rzz'

	def op_mat(self, params):
		return ZZPowMat(params[0])

#############################################################################

class SingleQubitRotationGate(Gate):
	def __init__(self, qubits):
		super().__init__(qubits)
		self.gate_string = 'SingleQubitRotation'
		self.ninput = 3
		self.nparams = 5
		self.initialize_offset()
		self.r = np.pi/2.

	def op_mat(self, params):
		U = ZPowMat(params[0])
		U = np.matmul(YPowMat(params[1]), U)
		U = np.matmul(ZPowMat(params[2]), U)
		U = np.matmul(YPowMat(params[3]), U)
		U = np.matmul(ZPowMat(params[4]), U)
		return U

	def format_params(self, ts):
		params = [ts[0], ts[1], ts[2], -ts[1], -ts[0]]
		return np.array(params)

	def get_differential_gates(self, diff_ind):
		if diff_ind == 2:
			offset_left = np.zeros(self.nparams)
			offset_left[2] = -np.pi/(4 * self.r)

			offset_right = np.zeros(self.nparams)
			offset_right[2] = np.pi/(4 * self.r)

			gleft = self.copy()
			gleft.set_offset(offset_left)
			gright = self.copy()
			gright.set_offset(offset_right)
			gates = [gright, gleft]
			weights = [self.r, -self.r]
			return gates, weights

		if diff_ind == 1:
			g_pos_right, g_pos_left, g_neg_right, g_neg_left = self.copy(), self.copy(), self.copy(), self.copy()

			offset_pos_right = np.array([0., np.pi/(4 * self.r), 0., 0., 0.])
			offset_pos_left = np.array([0., -np.pi/(4 * self.r), 0., 0., 0.])
			offset_neg_right = np.array([0., 0., 0., np.pi/(4 * self.r), 0.])
			offset_neg_left = np.array([0., 0., 0., -np.pi/(4 * self.r), 0.])
			
			g_pos_right.set_offset(offset_pos_right)
			g_pos_left.set_offset(offset_pos_left)
			g_neg_right.set_offset(offset_neg_right)
			g_neg_left.set_offset(offset_neg_left)

			gates = [g_pos_right, g_pos_left, g_neg_right, g_neg_left]
			weights = [self.r, -self.r, -self.r, self.r]
			return gates, weights

		if diff_ind == 0:
			g_pos_right, g_pos_left, g_neg_right, g_neg_left = self.copy(), self.copy(), self.copy(), self.copy()

			offset_pos_right = np.array([np.pi/(4 * self.r), 0., 0., 0., 0.])
			offset_pos_left = np.array([-np.pi/(4 * self.r), 0., 0., 0., 0.])
			offset_neg_right = np.array([0., 0., 0., 0., np.pi/(4 * self.r)])
			offset_neg_left = np.array([0., 0., 0., 0., -np.pi/(4 * self.r)])
			
			g_pos_right.set_offset(offset_pos_right)
			g_pos_left.set_offset(offset_pos_left)
			g_neg_right.set_offset(offset_neg_right)
			g_neg_left.set_offset(offset_neg_left)

			gates = [g_pos_right, g_pos_left, g_neg_right, g_neg_left]
			weights = [self.r, -self.r, -self.r, self.r]
			return gates, weights

#############################################################################

class IsingCouplingGate(Gate):
	def __init__(self, qubits):
		super().__init__(qubits)
		self.gate_string = 'IsingCoupling'
		self.ninput = 3
		self.nparams = 3
		self.initialize_offset()
		self.r = np.pi/2.

	def op_mat(self, params):
		U = XXPowMat(params[0])
		U = np.matmul(YYPowMat(params[1]), U)
		U = np.matmul(ZZPowMat(params[2]), U)
		return U

	def get_differential_gates(self, diff_ind):
		offset_left = np.zeros(self.nparams)
		offset_left[2] = -np.pi/(4 * self.r)

		offset_right = np.zeros(self.nparams)
		offset_right[2] = np.pi/(4 * self.r)

		gleft, gright = self.copy(), self.copy()
		gleft.set_offset(offset_left)
		gright.set_offset(offset_right)

		weights = [self.r, -self.r]
		gates = [gright, gleft]

		return gates, weights

#############################################################################

