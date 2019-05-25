import numpy as np

from cvrp.instance import *
from cvrp.nn import *


def roulette_wheel(probability_vector, random_state=None):

	if random_state is None:
		random_state = np.random

	wheel = np.cumsum(probability_vector/ np.sum(probability_vector))
	random_number = random_state.rand()

	index = next((-i for i, x in enumerate(wheel[::-1]) if x < random_number), -len(wheel))

	return index


def calculate_probability_vector(instance, colony, solution, feasible_steps=None):

	pheromone_vector = colony['pheromone_matrix'][solution['route'][-1]]
	distance_vector = instance['distance_matrix'][solution['route'][-1]]
	savings_vector = colony['savings_matrix'][solution['route'][-1]]

	tau = np.power(pheromone_vector, colony['alpha'])
	eta = np.power(np.array([1.])/distance_vector, colony['beta'])
	mu = np.power(savings_vector, colony['gama'])

	if feasible_steps is None:
		feasible_steps = find_feasible_steps(instance, solution, distance_vector)

	probability_vector = np.array([])

	if np.any(np.isinf(np.where(feasible_steps, eta, 0.))):
		probability_vector = np.where(np.isinf(eta), 1., 0.)
	elif np.all(mu == 0.):
		probability_vector = tau * eta
	else:
		probability_vector = tau * eta * mu

	probability_vector = np.where(feasible_steps, probability_vector, 0.)

	return probability_vector


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


def initialize_ants(instance):
    
	ants = []

	for i in range(instance['n']):
	    ant = initialize_solution(instance)
	    ant = move_to_city(instance, ant, i+1)
	    ants.append(ant)

	return ants


def initialize_ant_colony(
	    instance,
	    max_iter=50000, max_stable_iter=1000,
	    alpha=2, beta=5, gama=9, rho=.8, theta=80, sigma=3):

	pheromone_matrix = initialize_pheromone_matrix(instance)
	savings_matrix = calculate_savings_matrix(instance)

	ants = []
	ant_star = None
	current_iter = 0
	stable_iter = 0

	colony = dict(
	    max_iter=max_iter, max_stable_iter=max_stable_iter,
	    alpha=alpha, beta=beta, gama=gama, rho=rho, theta=theta, sigma=sigma,
	    pheromone_matrix=pheromone_matrix, savings_matrix=savings_matrix,
	    ants=ants, ant_star=ant_star, current_iter=current_iter, stable_iter=stable_iter)

	return colony


def ant_step(instance, colony, ant, random_state):
    
	feasible_steps = find_feasible_steps(instance, ant)

	if ~np.any(feasible_steps):
	    ant = move_to_depot(instance, ant)
	else:
	    probability_vector = calculate_probability_vector(instance, colony, ant)
	    city = roulette_wheel(probability_vector, random_state)

	    ant = move_to_city(instance, ant, city)

	return ant


def ant_route(instance, colony, ant=None, random_state=None):
    
	if ant is None:
	    ant = initialize_solution(instance)
	    
	if random_state is None:
	    random_state = np.random
	    
	while ~np.all(ant['visited_cities']):
	    ant = ant_step(instance, colony, ant, random_state)
	    
	ant = move_to_depot(instance, ant)

	return ant


def process_colony_routes(instance, colony, random_state=None):
    
	if random_state is None:
	    random_state = np.random

	colony['ants'] = initialize_ants(instance)

	for ant in colony['ants']:
	    ant.update(ant_route(instance, colony, ant, random_state))

	return colony


def colony_iteration(instance, colony, random_state=None):
    
	if random_state is None:
	    random_state = np.random
	    
	colony = process_colony_routes(instance, colony, random_state)
	colony['ants'] = sorted(colony['ants'], key=lambda ant: ant['route_length'])

	if colony['ant_star'] is None:
	    colony['ant_star'] = colony['ants'][0]
	elif colony['ants'][0]['route_length'] < colony['ant_star']['route_length']:
	    colony['ant_star'] = colony['ants'][0]
	    colony['stable_iter'] = 0
	else:
	    colony['stable_iter'] += 1
	    
	colony = update_pheromone_matrix(colony)
	    
	colony['current_iter'] += 1

	return colony


def solve_ant_colony(
		instance, ant_colony=None, random_state=None,
		log_history=False, report_iteration=None):
    
	if ant_colony is None:
	    ant_colony = initialize_ant_colony(instance)

	if random_state is None:
	    random_state = np.random
	   
	history = dict(best_solution=[], avg_solution=[], worst_solution=[])

	while (ant_colony['current_iter'] < ant_colony['max_iter']) and \
	      (ant_colony['stable_iter'] < ant_colony['max_stable_iter']):
	    
	    ant_colony = colony_iteration(instance, ant_colony, random_state)

	    if log_history:
	    	history['best_solution'].append(ant_colony['ant_star']['route_length'])
	    	history['avg_solution'].append(np.mean([ant['route_length'] for ant in ant_colony['ants']]))
	    	history['worst_solution'].append(np.max([ant['route_length'] for ant in ant_colony['ants']]))
	    
	    if report_iteration is not None:
	    	if ant_colony['current_iter'] % report_iteration == 0:
	        	print('current iter: {current_iter}\tstable iter: {stable_iter}'.format(**ant_colony))
	   
	if log_history:
		return ant_colony, history
	else:
		return ant_colony