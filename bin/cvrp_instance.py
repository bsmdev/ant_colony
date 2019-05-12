import numpy as np


def calculate_savings(city_vector, depot_vector):
    
	savings_vector = city_vector[0] + depot_vector - city_vector

	return savings_vector


def calculate_savings_matrix(instance):
    
	distance_matrix = instance['distance_matrix']
	savings_matrix = np.apply_along_axis(calculate_savings, 1, distance_matrix, distance_matrix[0])

	return savings_matrix


def initialize_solution(instance):

	solution = dict(
        route=[0],
        route_length = 0.,
        petal_length = 0.,
        cargo_delivered = 0.,
        visited_cities = np.array([True] + [False for i in range(instance['n'])])
    )

	return solution


def find_feasible_steps(instance, solution, distance_vector=None):
    
    if distance_vector is None:
        distance_vector = instance['distance_matrix'][solution['route'][-1]]
    
    distance_constraint = solution['petal_length'] + distance_vector + instance['d'] + instance['distance_matrix'][0] < instance['L']
    cargo_constraint = instance['demand_vector'] + solution['cargo_delivered'] < instance['Q']
    
    feasible_steps = ~solution['visited_cities'] & distance_constraint & cargo_constraint
    
    return feasible_steps


def move_to_city(instance, solution, city, distance_vector=None):

	if city < 0:
		city = instance['n'] + 1 + city

	if distance_vector is None:
		distance_vector = instance['distance_matrix'][solution['route'][-1]]

	solution['route_length'] += distance_vector[city]
	solution['petal_length'] += distance_vector[city] + instance['d']
	solution['cargo_delivered'] += instance['demand_vector'][city]

	solution['route'].append(city)
	solution['visited_cities'][city] = True

	return solution


def move_to_depot(instance, solution, distance_vector=None):
    
    if distance_vector is None:
        distance_vector = instance['distance_matrix'][solution['route'][-1]]
    
    solution['route_length'] += distance_vector[0]
    solution['petal_length'] = 0.
    solution['cargo_delivered'] = 0.
    solution['route'].append(0)
    
    return solution