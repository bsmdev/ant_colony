import numpy as np


def calculate_phi_matrix(instance):

	distance_matrix = instance['distance_matrix']

	depot_vector = distance_matrix[:,0]
	depot_column = np.reshape(depot_vector, (-1,1))

	phi_matrix = np.repeat(depot_column, len(depot_column), axis=1)
	phi_matrix = phi_matrix / depot_vector

	phi_matrix = np.where(phi_matrix < 1., -1./phi_matrix, phi_matrix)

	phi_matrix = np.where(np.isinf(phi_matrix), 1., phi_matrix)
	phi_matrix = np.where(np.isnan(phi_matrix), 1., phi_matrix)

	return phi_matrix


def calculate_phi_vector(instance, colony, solution):

	phi_vector = colony['phi_matrix'][solution['route'][-1]]
	q = solution['cargo_delivered'] / instance['Q'] - .5
	delta = colony['delta']

	if q > 0:
		phi_vector = np.where(
			phi_vector > 0.,
			np.power(phi_vector, delta*q),
			np.power(-1./phi_vector, delta*q))
	else:
		phi_vector = np.where(
			phi_vector > 0.,
			np.power(1./phi_vector, -delta*q),
			np.power(-phi_vector, -delta*q))

	return phi_vector


def make_colony_atomic(instance, colony, delta=5):

	phi_matrix = calculate_phi_matrix(instance)

	colony['phi_matrix'] = phi_matrix
	colony['delta'] = delta

	return colony