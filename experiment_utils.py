from arguments import args


#일반적인 edgelist -> orbit edge list
def make_orbit_edge(data_name = args.data):
    edge_path = f'data/{data_name}/edge_list.txt'
    orbit_edge_path = f'data/{data_name}/orbit_edge_list.txt'

    edge = open(edge_path, 'r')
    orbit_edge = open(orbit_edge_path, 'w')
    for line in edge:
        node1, node2 = line.strip().split(' ')
        if node1 < node2:
            orbit_edge.write(f'{node1}\t{node2}\n')

