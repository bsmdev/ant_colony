import numpy as np


def distances(x, y, x_vector, y_vector):
    
    delta_x = x_vector - x
    delta_y = y_vector - y
    
    distances = np.sqrt(delta_x*delta_x + delta_y*delta_y)
    
    return distances


def create_instance(file_contents):
    
    lines = file_contents.split('\n')
    n, Q, L, d = np.array(lines[0].split(' ')[1:]).astype(int)
    
    x_vector = np.array([])
    y_vector = np.array([])
    demand_vector = np.array([])
    
    x, y = np.array(lines[1].split(' ')[1:]).astype(int)
    x_vector = np.append(x_vector, x)
    y_vector = np.append(y_vector, y)
    demand_vector = np.append(demand_vector, 0)
    
    for i in range(n):
        x, y, demand = np.array(lines[i+2].split(' ')[1:]).astype(int)
        x_vector = np.append(x_vector, x)
        y_vector = np.append(y_vector, y)
        demand_vector = np.append(demand_vector, demand)
        
    distance_matrix = []
    for i in range(n+1):
        x = x_vector[i]
        y = y_vector[i]
        distance_matrix.append(distances(x, y, x_vector, y_vector))
    distance_matrix = np.array(distance_matrix).reshape(n+1, n+1)
    
    instance = dict(
        n=n, Q=Q, L=L, d=d,
        distance_matrix=distance_matrix,
        demand_vector=demand_vector,
        x=x_vector, y=y_vector)
    
    return instance


def load_instance(path):

	f = open(path, 'r')
	contents = f.read()
	instance = create_instance(contents)
	f.close()

	return instance