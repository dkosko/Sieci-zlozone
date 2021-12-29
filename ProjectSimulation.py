import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import get_backend
import numpy as np
import scipy.stats as st
from math import log
import random as r
import copy


class Osoba:
    def __init__(self, index, chory):
        self.i = index
        self.chory = chory



def G_statistics(G):
    print('stopnie: ', nx.degree(G))
    print('gęstość: ', nx.density(G))
    print('bliskość: ', nx.closeness_centrality(G))
    print('pośrednictwo: ', nx.betweenness_centrality(G))
    print('średnica: ', nx.diameter(G))
    for x in nx.connected_components(G):
        print('skladowa spójna: ', x)
    print('spójność krawędziowa: ', nx.edge_connectivity(G))
    print('spójność wierszchołkowa: ', nx.node_connectivity(G))


def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)

def plot_color_graph(G, color_map, t=1):
    f, ax = plt.subplots()
    move_figure(f, 900, 100)
    pos = nx.spring_layout(G, k=0.9, seed=111)
    nx.draw(G, node_color=color_map, with_labels=True, pos=pos, )
    plt.show()
    # plt.show(block=False)
    # plt.pause(t)
    # plt.close()


def plot_weighted_graph(G, color_map, t=1):
    f, ax = plt.subplots()
    move_figure(f, 900, 100)
    pos = nx.spring_layout(G, k=0.9, seed=111)
    nx.draw(G, node_color=color_map, with_labels=True, pos=pos, )
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()
    # plt.show(block=False)
    # plt.pause(t)
    # plt.close()


def get_neighbours(G,node):
    return list(G[node].keys())

def if_red_neighbour(G, node, color_map):
    neighbours = get_neighbours(G, node)
    for neighbour in neighbours:
        if color_map[neighbour] == 'red':
            return 1

def simulate_1(n=15, do_plots=1):
    G = nx.barabasi_albert_graph(n, 1, seed=111)
    color_map = []
    for i in range(n):
        color_map.append('blue')
    if do_plots: plot_color_graph(G, color_map)
    color_map[0] = 'red'
    if do_plots: plot_color_graph(G, color_map)


    end_colmap = []
    for i in range(n):
        end_colmap.append('red')

    count = 0
    color_map_1 = copy.deepcopy(color_map)
    for i in range(nx.diameter(G)):
        count += 1
        for j in range(n):
            if color_map[j] == 'blue' and if_red_neighbour(G, j,color_map):
                color_map_1[j] = 'red'
        color_map = copy.deepcopy(color_map_1)
        if color_map == end_colmap:
            break
        if do_plots: plot_color_graph(G, color_map)


    if do_plots: plot_color_graph(G, color_map)

    print('Liczba iteracji= ', count)

def simulate_2(n=15, do_plots = 1):
    G = nx.barabasi_albert_graph(n, 1, seed=111)
    color_map = []
    for i in range(n):
        color_map.append('blue')
    if do_plots: plot_color_graph(G, color_map)
    color_map[0] = 'red'
    if do_plots: plot_color_graph(G, color_map)


    end_colmap = []
    for i in range(n):
        end_colmap.append('red')

    count = 0
    color_map_1 = copy.deepcopy(color_map)
    for i in range(100):
        count += 1
        for j in range(n):
            if color_map[j] == 'blue' and if_red_neighbour(G, j,color_map):
                if r.uniform(0,1) > 0.5:
                    color_map_1[j] = 'red'
        color_map = copy.deepcopy(color_map_1)
        if color_map == end_colmap:
            break
        if do_plots: plot_color_graph(G, color_map)


    if do_plots: plot_color_graph(G, color_map)

    print('Liczba iteracji= ', count)


def network_generator(num_cliques = 30, num_edges = 80):
    def add_clique(G):
        # 5% - 1os , 25% - 2os, 40% - 3os, 25% - 4os, 5% - 5os
        clique_size = r.choices([1, 2, 3, 4, 5], weights=[5, 12, 13, 6, 1])[0]
        nodes = list(G.nodes)
        if nodes == []:
            last_node = 0
        else:
            last_node = nodes[-1] + 1

        Clique = nx.complete_graph(range(last_node, last_node+clique_size))

        G.add_edges_from(Clique.edges)
        for edge in Clique.edges:
            G[edge[0]][edge[1]]['weight'] = round(r.uniform(0.5, 0.9),2)
        return G


    def add_edge(G):
        edges = list(G.edges)
        nodes = list(G.nodes)
        n1 = r.choice(nodes)
        nodes.remove(n1)
        n2 = r.choice(nodes)
        if (n1,n2) in edges:
            add_edge(G)
        else:
            G.add_edge(n1, n2, weight= round(r.uniform(0.1, 0.5),2))
        return G


    G = nx.Graph()
    for i in range(num_cliques):
        G = add_clique(G)
    #plot_color_graph(G, color_map=['blue'] * G.number_of_nodes(), t=2)
    print(G.number_of_nodes(),' nodes, ',G.number_of_edges(), ' edges')
    for i in range(num_edges):
        G = add_edge(G)
    #plot_color_graph(G, color_map=['blue'] * G.number_of_nodes(), t=5)
    print(G.number_of_nodes(),' nodes, ',G.number_of_edges(), ' edges')

    while nx.node_connectivity(G) == 0:
        G = add_edge(G)
    print(G.number_of_nodes(), ' nodes, ', G.number_of_edges(), ' edges')
    #plot_color_graph(G, color_map=['blue'] * G.number_of_nodes(), t=5)

    return G

if __name__ == "__main__":
    # simulate_1(15, 0)
    # simulate_2(15, 1)
    # G = network_generator()
    # nx.write_edgelist(G, 'Graph.gz')
    G = nx.read_edgelist('Graph.gz')
    #print(G.number_of_nodes(), ' nodes, ', G.number_of_edges(), ' edges')
    #G_statistics(G)
    plot_weighted_graph(G, color_map=['blue'] * G.number_of_nodes())


    osoby = []
    for i in range(0, nx.number_of_nodes(G)):
        osoba = Osoba(i, 0)
        osoby.append(osoba)


