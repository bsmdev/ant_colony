import numpy as np

from cvrp.instance import *
from ant_colony.desireability import *


def calculate_desireability_vector(instance, colony, solution, feasible_steps=None):

	desireability_vector = colony['desireability_matrix'][solution['route'][-1]]

	if feasible_steps is None:
		feasible_steps = find_feasible_steps(instance, solution, distance_vector)

	desireability_vector = np.where(feasible_steps, desireability_vector, 0.)

	if np.any(np.isnan(desireability_vector)):
		desireability_vector = np.where(np.isnan(desireability_vector), 1., 0.)

	return desireability_vector


def spin_roulette(desireability_vector, random_state=None):

	if random_state is None:
		random_state = np.random
  
	roulette_strip = np.cumsum(desireability_vector)
	spin = random_state.rand() * np.max(roulette_strip)

	winner = np.where(spin < roulette_strip)[0][0]

	return winner


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
	    alpha=2, beta=5, gamma=9, rho=.8, theta=80, sigma=3):
	
	pheromone_matrix = initialize_pheromone_matrix(instance)
	eta_mu_matrix = calculate_eta_mu_matrix(instance, beta, gamma)

	ants = []
	ant_star = None
	current_iter = 0
	stable_iter = 0

	colony = dict(
	    max_iter=max_iter, max_stable_iter=max_stable_iter,
	    alpha=alpha, beta=beta, gamma=gamma, rho=rho, theta=theta, sigma=sigma,
	    pheromone_matrix=pheromone_matrix, eta_mu_matrix=eta_mu_matrix,
	    ants=ants, ant_star=ant_star, current_iter=current_iter, stable_iter=stable_iter)

	return colony


def ant_step(instance, colony, ant, random_state=None):

	if random_state is None:
	    random_state = np.random
    
	feasible_steps = find_feasible_steps(instance, ant)

	if ~np.any(feasible_steps):
		ant = move_to_depot(instance, ant)
	else:
	    desireability_vector = calculate_desireability_vector(instance, colony, ant, feasible_steps)
	    city = spin_roulette(desireability_vector, random_state)
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

	colony = update_desireability_matrix(colony)

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


def solve(
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