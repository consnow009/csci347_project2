# Project 2, Part 2 Code
# CSCI 347: Data Mining
# Connor Lowe
# 26 February 2021

import test_case_graphs as tcg
import networkx as nx
import numpy as np


# Function for number of vertices in a graph
def num_vertices(graph):
    # Uses list comprehension to convert list of tuples to regular list
    out = [item for x in graph for item in x]
    # Extracts unique values from list
    unique = set(out)
    # Counts the number of unique values in list, which is the number of vertices
    count = len(unique)
    return count


# Function for finding the degree of a vertex
def deg_vertex(graph, vertex):
    # Converts tuples to regular list
    regular_list = [item for x in graph for item in x]
    # Counts the number of times the given vertex appears in the list
    count = 0
    for item in regular_list:
        if item == vertex:
            count = count + 1
    return count


# Helper function to get number number of edges from subgraph created from specified nodes
def subgraph_edges(nodes, graph):
    # Creates an empty graph
    G = nx.Graph()
    # Uses edge list to create graph
    G.add_edges_from(graph)
    # Creates a subgraph from the specified nodes
    induced_subgraph = G.subgraph(nodes)
    # Gets the number of edges in the subgraph
    number_of_edges = nx.number_of_edges(induced_subgraph)
    return number_of_edges


# Helper function to generate edge list into graph
def generate_graph(edgelist):
    G = nx.Graph()
    G.add_edges_from(edgelist)
    return G


# Function for finding the clustering coefficient of a vertex
# ref: https://www.geeksforgeeks.org/python-find-the-tuples-containing-the-given-element-from-a-list-of-tuples/1
def clustering_coef(graph, vertex):
    # Obtains a list of all the neighbors of the given vertex
    filtered_list = list(filter(lambda x: vertex in x, graph))
    # Converts the list of tuples into a regular list
    regular_list = [item for x in filtered_list for item in x]
    # Remove the vertex value from the regular list to get the list of just the neighbors
    neighbors = list(filter(vertex.__ne__, regular_list))
    # Calls a helper function that generates a graph from the edge list and creates a subgraph
    # of neighbors and then finds the number of edges amongst the subgraph of neighbors
    total_edges_actual = subgraph_edges(neighbors, graph)
    # Gets the number of neighbors
    number_of_neighbors = len(neighbors)
    # Calculates the number of total edges possible by (n-1)*(n/2) where n is the number of vertices
    total_edges_possible = (number_of_neighbors - 1) * (number_of_neighbors / 2)
    # Calculates the clustering coefficient by dividing
    # the number of actual edges by the total number of possible edges
    clustering_coefficient = total_edges_actual / total_edges_possible
    return clustering_coefficient


# Function for finding the betweenness centrality of a vertex
def betweenness_centrality(edgelist, vertex):
    values = []
    betweenness = 0
    size = num_vertices(edgelist)
    # Generate a graph from the given edgelist using a helper function
    graph = generate_graph(edgelist)

    # To determine if edgelist starts with 0 vs 1
    if edgelist[0][0] == 1:
        vertex_list = list(range(1, 1 + size))
    else:
        vertex_list = list(range(0, size))

    # Removes the specified vertex from the vertex list since it
    # won't be used in making a list of all possible vertex pairs
    vertex_list.remove(vertex)

    # Create a list of all the vertex pairs excluding x,x pairs (i.e. 1,1 or 2,2 etc.)
    for s in range(len(vertex_list)):
        for t in range(len(vertex_list)):
            if s != t:
                values = values + [[vertex_list[s], vertex_list[t]]]

    # Get rid of mirrored duplicates (i.e. [[0,1],[1,0]] -> [[0,1]])
    s = set()
    out = []
    for i in values:
        t = tuple(i)
        if t in s or tuple(reversed(t)) in s:
            continue
        s.add(t)
        out.append(i)

    # Extract the columns from the nested list so we have a list of every
    # possible node pair with no repeats and no x, x node pairs
    x_bar = [i[0] for i in out]
    y_bar = [i[1] for i in out]

    # Simultaneously iterate through the separate columns representing all vertex pairs and find all shortest paths
    for (x, y) in zip(x_bar, y_bar):
        count = 0

        # Get a list of all the shortest paths for every pair of nodes
        list_shortest_paths = list([p for p in nx.all_shortest_paths(graph, source=x, target=y)])

        # Get the total number of shortest paths
        number_of_shortest_path = len(list_shortest_paths)

        # Flatten the nested list into a list of single elements
        flat_list = [item for sublist in list_shortest_paths for item in sublist]

        # Search for the number of occurrences of the betweenness node
        for i in flat_list:
            if i == vertex:
                count = count + 1

        # Calculate betweenness centrality
        betweenness = betweenness + (count / number_of_shortest_path)
    return betweenness


def average_shortest_path_length(edgelist):
    sum_of_shortest_paths = 0
    values = []
    size = num_vertices(edgelist)
    n = size * (size - 1)

    # Generate a graph from the given edgelist using a helper function
    graph = generate_graph(edgelist)

    # To determine if edgelist starts with 0 vs 1
    if edgelist[0][0] == 1:
        vertex_list = list(range(1, 1 + size))
    else:
        vertex_list = list(range(0, size))

    # Create a list of all the vertex pairs excluding x,x pairs (i.e. 1,1 or 2,2 etc.)
    for s in range(len(vertex_list)):
        for t in range(len(vertex_list)):
            if s != t:
                values = values + [[vertex_list[s], vertex_list[t]]]

    # Extract the columns from the nested list so we have a list of every
    # possible node pair with no x, x node pairs
    x_bar = [i[0] for i in values]
    y_bar = [i[1] for i in values]

    # Simultaneously iterate through the separate columns representing all vertex pairs and find all shortest paths
    for (x, y) in zip(x_bar, y_bar):
        # Get a list of all the shortest paths for every pair of nodes
        shortest_path = nx.shortest_path(graph, source=x, target=y)
        sum_of_shortest_paths = sum_of_shortest_paths + len(shortest_path) - 1

    # Calculate the average shortest path length
    average = sum_of_shortest_paths / n
    return average


# Function for creating an adjacency matrix
def adjacency_matrix(edgelist):
    # Get the size of the edgelist (number of nodes)
    nodes = num_vertices(edgelist)
    # Create an array of all zeros based on the number of nodes
    matrix = np.zeros((nodes, nodes))
    # Use the vertices as coordinate to iterate thru the array
    x_bar = [i[0]-1 for i in edgelist]
    y_bar = [i[1]-1 for i in edgelist]

    # Using the coordinates change a zero to a one
    for (x, y) in zip(x_bar, y_bar):
        matrix[x][y] = 1
        matrix[y][x] = 1
    return matrix


# Main
print("Number of Vertices:", num_vertices(tcg.graph_8))
print("Degree of Vertex:", deg_vertex(tcg.graph_7, 27))
print("Clustering Coefficient:", clustering_coef(tcg.graph_8, 46))
print("Betweenness Centrality:", betweenness_centrality(tcg.graph_0, 5))
print("Average Shortest Path Length:", average_shortest_path_length(tcg.graph_8))
print("Adjacency Matrix:\n", adjacency_matrix(tcg.graph_6))
