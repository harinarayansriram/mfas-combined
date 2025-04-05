import pandas as pd
import networkx as nx
from networkx.algorithms.cycles import simple_cycles
import gurobipy as gp
from gurobipy import GRB
import sys
from datetime import datetime
import time

FileNameHead="IP"
# Read the CSV file and create a directed graph
def read_graph_from_csv(file_path):
    #df = pd.read_csv(file_path)
    df=pd.read_csv(file_path, header=None, names=['source', 'destination', 'weight'])
    G = nx.DiGraph()
    for index, row in df.iterrows():
        G.add_edge(row['source'], row['destination'], weight=int(row['weight']))
        #print("source=",row['source'], "dest=",row['destination'], "weight=",row['weight'])
    return G

# Find all directed cycles in the graph
def find_cycles(G):
    return list(simple_cycles(G))
    #  next_n_cycles = list(itertools.islice(all_cycles, start, start + number))

def build_cycle_graph(original_graph):
    """
    Build a new directed graph based on cycles from the original directed graph.
    
    Parameters:
    original_graph (networkx.DiGraph): The original directed graph.
    
    Returns:
    networkx.DiGraph: A new directed graph containing edges from cycles.
    """
    cycle_graph = nx.DiGraph()
    
    # Get all cycles in the original graph
    cycles = nx.simple_cycles(original_graph)

    # Add edges from each cycle to the new graph
    for cycle in cycles:
        # Add edges of the cycle to the new graph
        for i in range(len(cycle)):
            source = cycle[i]
            destination = cycle[(i + 1) % len(cycle)]  # wrap around to the start
            # Check if the edge exists in the original graph
            if original_graph.has_edge(source, destination):
                weight = original_graph[source][destination]['weight']
                cycle_graph.add_edge(source, destination, weight=weight)

    return cycle_graph



# Formulate the integer programming problem
def solve_ip_for_minimum_feedback_arc_set(G, cycles):
    model = gp.Model("min_feedback_arc_set")
    model.setParam('OutputFlag', 0)  # Silent mode

    # Create binary variables for each edge
    edge_vars = {}
    for u, v, data in G.edges(data=True):
        edge_vars[(u, v)] = model.addVar(vtype=GRB.BINARY, obj=data['weight'])

    # Add constraints for each cycle
    for cycle in cycles:
        cycle_edges = [(cycle[i], cycle[(i+1) % len(cycle)]) for i in range(len(cycle))]
        model.addConstr(gp.quicksum(edge_vars[edge] for edge in cycle_edges) >= 1)

    # Optimize the model
    model.optimize()

    # Get the edges to be removed
    removed_edges = [edge for edge, var in edge_vars.items() if var.x > 0.5]
    return removed_edges


# Calculate the total weight of the edges
def total_edge_weight(G):
    return sum(data['weight'] for u, v, data in G.edges(data=True))

# Calculate the weight of the removed edges
def removed_edge_weight(G, removed_edges):
    return sum(G[u][v]['weight'] for u, v in removed_edges)

# Identify the removed edges and construct a DAG
def construct_dag(G, removed_edges):
    G_dag = G.copy()
    G_dag.remove_edges_from(removed_edges)
    return G_dag

# Relabel the vertices in the DAG
def relabel_dag(G_dag):
    topological_order = list(nx.topological_sort(G_dag))
    mapping = {node: i for i, node in enumerate(topological_order)}
    G_dag = nx.relabel_nodes(G_dag, mapping)
    return G_dag, mapping

# Write the original vertex ID and its relative order to a file
def write_relabelled_nodes_to_file(mapping, output_file):
    with open(output_file, 'w') as f:
        for node, order in mapping.items():
            f.write(f"{node},{order}\n")

# Main function to perform all steps
def process_graph(file_path):

    print("read file")
    starttime = time.time()
    G = read_graph_from_csv(file_path)
    endtime = time.time()
    executiontime=endtime-starttime
    current_time = datetime.now()
    time_string = current_time.strftime("%Y%m%d_%H%M%S")
    # Create the file name using the formatted time string
    oneoutput_file = f"{FileNameHead}-{time_string}.txt"
    with open(oneoutput_file, 'w') as f:
            f.write(f"reading file {file_path} takes {executiontime} seconds \n")


    starttime = time.time()
    cycles = find_cycles(G)
    endtime = time.time()
    executiontime=endtime-starttime
    with open(oneoutput_file, 'a') as f:
            f.write(f"finding cycles in file {file_path} uses {executiontime} seconds \n")
            f.write(f"The total number of cycles is {len(cycles)}\n")


    starttime = time.time()
    cycle_graph = build_cycle_graph(G)
    endtime = time.time()
    executiontime=endtime-starttime
    current_time = datetime.now()
    with open(oneoutput_file, 'a') as f:
            f.write(f"building cycle graph in file {file_path} uses {executiontime} seconds \n")

    starttime = time.time()
    removed_edges = solve_ip_for_minimum_feedback_arc_set(cycle_graph, cycles)
    endtime = time.time()
    executiontime=endtime-starttime
    current_time = datetime.now()
    total_weight = total_edge_weight(G)
    removed_weight = removed_edge_weight(G, removed_edges)
    percentage_removed = (removed_weight / total_weight) * 100
    with open(oneoutput_file, 'a') as f:
        f.write(f"solving the feedback arc set in cycle graph of file {file_path} uses {executiontime} seconds \n")
        f.write(f"Total Edge Weight:{total_weight}, Total number of Edges= {G.number_of_edges()}\n")
        f.write(f"Removed Edge Weight: {removed_weight} Removed number of Edges={len(removed_edges)}\n")
        f.write(f"Percentage of Removed Edges Weight: {percentage_removed}\n")
        f.write(f"All removed edges are as follows \n")
        for u, v in removed_edges:
            f.write(f"{u},{v},{G[u][v]['weight']}\n")

    starttime = time.time()
    G_dag = construct_dag(G, removed_edges)
    endtime = time.time()
    executiontime=endtime-starttime
    current_time = datetime.now()
    with open(oneoutput_file, 'a') as f:
            f.write(f"construct DAG graph in  file {file_path} uses {executiontime} seconds \n")

    starttime = time.time()
    G_dag_relabelled, mapping = relabel_dag(G_dag)
    dag_weight = total_edge_weight(G_dag)
    endtime = time.time()
    executiontime=endtime-starttime
    current_time = datetime.now()
    with open(oneoutput_file, 'w') as f:
            f.write(f"Topological sort in  generated DAG graph of  file {file_path} uses {executiontime} seconds \n")
            f.write(f"The DAG graph weight percentage is {dag_weight/total_weight * 100} \n")


    starttime = time.time()
    current_time = datetime.now()
    time_string = current_time.strftime("%Y%m%d_%H%M%S")
    #Create the file name using the formatted time string
    output_file = f"{FileNameHead}-07-{file_path}-Relabel-{time_string}.csv"
    write_relabelled_nodes_to_file(mapping, output_file)
    endtime = time.time()
    executiontime=endtime-starttime
    current_time = datetime.now()
    with open(oneoutput_file, 'w') as f:
            f.write(f"write label of file {file_path} uses {executiontime} seconds \n")

file_path = sys.argv[1]

process_graph(file_path)

