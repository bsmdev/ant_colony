import numpy as np

from .instance import *


def nearest_neighbor_step(instance, solution):
    
    distance_vector = instance['distance_matrix'][solution['route'][-1]]
    
    feasible_steps = find_feasible_steps(instance, solution, distance_vector)
    
    if ~np.any(feasible_steps):
        solution = move_to_depot(instance, solution, distance_vector)
    else:
        manipulated_distances = np.where(feasible_steps, distance_vector, instance['L'])
        min_distance = np.min(manipulated_distances)
        nearest_neighbor = np.where(manipulated_distances==min_distance)[0][0]
        
        solution = move_to_city(instance, solution, nearest_neighbor, distance_vector)
    
    return solution


def nearest_neighbor_route(instance):
    
    solution  = initialize_solution(instance)

    while ~np.all(solution['visited_cities']):
        solution = nearest_neighbor_step(instance, solution)

    solution = move_to_depot(instance, solution)

    return solution