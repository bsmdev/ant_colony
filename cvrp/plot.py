import plotly.graph_objs as go


def create_cities_tracer(instance,
                         depot_color='#1f77b4', city_color='#777'):
    
    color = [depot_color] + [city_color for i in range(instance['n'])]
    
    cities_tracer = dict(
        type='scatter', mode='markers',
        x=instance['x'],
        y=instance['y'],
        marker=dict(color=color),
        showlegend=False)
    
    return cities_tracer


def create_route_tracer(instance, solution):
    
    route_tracer = dict(
        type='scatter', mode='lines',
        x=instance['x'][solution['route']],
        y=instance['y'][solution['route']],
        line=dict(color='rgba(0,0,0,.2)'),
        showlegend=False)
    
    return route_tracer


def create_instance_figure(instance):
    
    tracers = []
    
    cities_tracer = create_cities_tracer(instance)
    tracers.append(cities_tracer)
    
    layout = dict(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        hovermode='closest')
    
    figure = go.Figure(data=tracers, layout=layout)
    return figure


def create_solution_figure(instance, solution):
    
    tracers = []
    
    route_tracer = create_route_tracer(instance, solution)
    tracers.append(route_tracer)
    
    cities_tracer = create_cities_tracer(instance)
    tracers.append(cities_tracer)
    
    layout = dict(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        hovermode='closest')
    
    figure = go.Figure(data=tracers, layout=layout)
    return figure