import matplotlib.pyplot as plt
import numpy as np

from utils import *
from circuit import *
from hamiltonian import *
from vqt import *

####################################################
###########################################################################
###########################################################################

def test_vqt():
	nqubits = 2
	J = -1.0
	hx = 0.3
	hz = 0.1
	H = gen_heisenberg1d_hamiltonian(nqubits, J = J, hx = hx, hz = hz)

	beta = 2.5

	nlayers = 8
	factorized = False ## type of latent space distribution to use ##
	circ = generate_circuit(nqubits, nlayers, factorized = factorized)

	##### If want to use noise:
	# noise = 8.0E-4
	# nmeas = 10
	##### Noise-free parameters
	noise = None
	nmeas = 1

	vqt = VQTEngine(circ, H, beta, noise, nmeas)
	rho_thermal = vqt.rho_thermal
	F_thermal = get_free_energy(rho_thermal, beta, H)

	np.random.seed(127)

	print("Thermal Free Energy: {}".format(F_thermal))
	vqt.run()
	hist = vqt.loss_history
	plt.plot(hist)
	plt.show()
	plt.cla()
	plt.clf()

	rho_vqt = vqt.get_circuit_rho()
	rho_thermal = vqt.rho_thermal
	F_thermal = get_free_energy(rho_thermal, beta, H)
	print("Thermal Free Energy: {}".format(F_thermal))
	F_vqt = get_free_energy(rho_vqt, beta, H)
	print("VQT Free Energy: {}".format(F_vqt))
	fid = get_fidelity(rho_thermal, rho_vqt)
	td = get_trace_distance(rho_thermal, rho_vqt)
	print('Fidelity = {}'.format(fid))
	print('Trace Dist = {}'.format(td))





####################################################

def main():
	test_vqt()


####################################################



if __name__ == "__main__":
	main()






