import numpy as np

from cvrp.instance import *
from cvrp.nn import *


def calculate_savings(city_vector, depot_vector):
    
	savings_vector = city_vector[0] + depot_vector - city_vector

	return savings_vector


def calculate_savings_matrix(instance):
    
	distance_matrix = instance['distance_matrix']

	savings_matrix = np.apply_along_axis(calculate_savings, 1, distance_matrix, distance_matrix[0])
	savings_matrix = np.where(
		savings_matrix < np.finfo(np.float64).eps,
		np.ones(np.shape(savings_matrix)),
		savings_matrix)

	return savings_matrix


def initialize_pheromone_matrix(instance):

	nn_solution = nearest_neighbor_route(instance)

	pheromone_matrix = np.ones(instance['distance_matrix'].shape)
	pheromone_matrix *= instance['n'] / nn_solution['route_length']

	return pheromone_matrix


def update_pheromone_matrix(colony):
    
	avg_distance = np.mean([ant['route_length'] for ant in colony['ants']])
	colony['pheromone_matrix'] *= colony['rho'] + colony['theta'] / avg_distance

	for rank, ant in enumerate(colony['ants'][:colony['sigma']-1]):
		for i, _ in enumerate(ant['route'][:-1]):
			colony['pheromone_matrix'][ant['route'][i]][ant['route'][i+1]] += (colony['sigma'] - rank-1) / ant['route_length']

	ant_star = colony['ant_star']
	for i, _ in enumerate(ant_star['route'][:-1]):
		colony['pheromone_matrix'][ant_star['route'][i]][ant_star['route'][i+1]] += colony['sigma'] / ant_star['route_length']

	return colony


def calculate_eta_mu_matrix(instance, beta, gamma):

	distance_matrix = instance['distance_matrix']
	savings_matrix = calculate_savings_matrix(instance)

	inv_distance_matrix = np.array([1.]) / distance_matrix
	inv_distance_matrix = np.where(np.isinf(inv_distance_matrix), np.nan, inv_distance_matrix)
	eta_mu_matrix = np.power(inv_distance_matrix, beta) * np.power(savings_matrix, gamma)

	return eta_mu_matrix


def update_desireability_matrix(colony):

	eta_mu_matrix = colony['eta_mu_matrix']
	pheromone_matrix = colony['pheromone_matrix']
	alpha = colony['alpha']

	desireability_matrix = eta_mu_matrix * np.power(pheromone_matrix, alpha)
	colony['desireability_matrix'] = desireability_matrix

	return colony
