import numpy as np


def solution_report(instance, solution):
    
    # DEPOT AT START AND END OF ROUTE #
    test_description = 'Depot at the start and end of the route'
    if (solution['route'][0] == 0) and (solution['route'][-1] == 0):
        print('PASSED:\t{}'.format(test_description))
    else:
        print('FAILED:\t{}'.format(test_description))
    ###
    
    route = np.array(solution['route'])
    cities_visited = np.array(route[route != 0])
    
    unique, counts = np.unique(cities_visited, return_counts=True)
    
    # ALL CITIES VISITED #
    test_description = 'All cities visited ({} of {})'.format(len(unique), instance['n'])
    if len(unique) == instance['n']:
        print('PASSED:\t{}'.format(test_description))
    else:
        print('FAILED:\t{}'.format(test_description))
    ###
    
    # ALL CITIES VISITED ONCE #
    test_description = 'Cities visited only once'
    if np.all(counts == 1):
        print('PASSED:\t{}'.format(test_description))
    else:
        print('FAILED:\t{}'.format(test_description))
    ###
    
    route = str(solution['route'][1:-1])[1:-1]
    route = route.split(', 0, ')
    
    petals = []
    for petal in route:
        
        cities = np.array(petal.split(', ')).astype(int)
        
        length = instance['distance_matrix'][0][cities[0]]
        for i, c in enumerate(cities[:-1]):
            length += instance['distance_matrix'][cities[i]][cities[i+1]]
            length += instance['d']
        length += instance['distance_matrix'][cities[-1]][0]
        
        cargo = 0.
        for c in cities:
            cargo += instance['demand_vector'][c]
        
        petals.append(dict(cities=cities, length=length, cargo=cargo))
    
    for i, petal in enumerate(petals):
        
        # PETAL LENGTH SMALLER THAN DISTANCE CONSTRAINT #
        test_description = 'Petal {} length shorter than distance constraint ({:.1f} < {})'.format(i, petal['length'], instance['L'])
        if petal['length'] < instance['L']:
            print('PASSED:\t{}'.format(test_description))
        else:
            print('FAILED:\t{}'.format(test_description))
        ###
        
    for i, petal in enumerate(petals):
            
        # PETAL DEMAND SMALLER THAN CARGO CONSTRAINT #
        test_description = 'Petal {} demand smaller than cargo constraint ({:.1f} < {})'.format(i, petal['cargo'], instance['Q'])
        if petal['cargo'] < instance['Q']:
            print('PASSED:\t{}'.format(test_description))
        else:
            print('FAILED:\t{}'.format(test_description))
        ###