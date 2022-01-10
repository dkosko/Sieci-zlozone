import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from ProjectSimulation import network_generator

def SIR(G, Nb_inf_init, Gamma, N, T):
    """ function that runs a simulation of an SIR model on a network.
    Args:
        Gamma(float): recovery rate
        Beta(float): infection probability
        Rho(float): initial fraction of infected individuals
        N(int): number of agents (nodes)
        T(int): number of time steps simulated
    """
    A = nx.to_numpy_array(G)
    # setting initial conditions
    s = np.zeros(T)
    inf = np.zeros(T)
    r = np.zeros(T)
    inf[0] = Nb_inf_init
    s[0] = N - Nb_inf_init
    """Make a graph with some infected nodes."""
    for u in G.nodes():
        G.nodes[u]["state"] = 0
        G.nodes[u]["TimeInfected"] = 0
        G.nodes[u]["noeux_associes"] = [n for n in G.neighbors(u)]

    init = random.sample(list(G.nodes()), Nb_inf_init)
    for u in init:
        G.nodes[u]["state"] = 1
        G.nodes[u]["TimeInfected"] = 1
    # running simulation
    for t in range(1, T):
        s[t] = s[t - 1]
        inf[t] = inf[t - 1]
        r[t] = r[t - 1]
        # Check which persons have recovered
        for u in G.nodes:
            # if infected
            if G.nodes[u]["state"] == 1:
                if G.nodes[u]["TimeInfected"] < Gamma:
                    G.nodes[u]["TimeInfected"] += 1
                else:
                    G.nodes[u]["state"] = 2  # "recovered"
                    r[t] += 1
                    inf[t] += -1
        # check contagion
        for u in G.nodes:
            # if susceptible
            if G.nodes[u]["state"] == 0:
                nb_friend_infected = [G.nodes[n]["state"] == 1 for n in G.nodes[u]["noeux_associes"]].count(True)
                # print(nb_friend_infected)
                for n in G.nodes[u]["noeux_associes"]:
                    if G.nodes[n]["state"] == 1:  # if friend is infected
                        # with HM infect
                        if random.uniform(0,1) < G[u][n]['weight']:
                            G.nodes[u]["state"] = 1
                            inf[t] += 1
                            s[t] += -1
                            break

    return s, inf, r

def SIR_norm(G, Nb_inf_init, Gamma, N, T):
    """ function that runs a simulation of an SIR model on a network.
    Args:
        Gamma(float): recovery rate
        Beta(float): infection probability
        Rho(float): initial fraction of infected individuals
        N(int): number of agents (nodes)
        T(int): number of time steps simulated
    """
    A = nx.to_numpy_array(G)
    # setting initial conditions
    s = np.zeros(T)
    inf = np.zeros(T)
    r = np.zeros(T)
    inf[0] = Nb_inf_init
    s[0] = N - Nb_inf_init
    """Make a graph with some infected nodes."""
    for u in G.nodes():
        G.nodes[u]["state"] = 0
        G.nodes[u]["TimeInfected"] = 0
        G.nodes[u]["noeux_associes"] = [n for n in G.neighbors(u)]

    init = random.sample(list(G.nodes()), Nb_inf_init)
    for u in init:
        G.nodes[u]["state"] = 1
        G.nodes[u]["TimeInfected"] = 1
    # running simulation
    for t in range(1, T):
        s[t] = s[t - 1]
        inf[t] = inf[t - 1]
        r[t] = r[t - 1]
        # Check which persons have recovered
        for u in G.nodes:
            # if infected
            if G.nodes[u]["state"] == 1:
                if G.nodes[u]["TimeInfected"] < round(np.random.normal(Gamma,3,1)[0]):
                    G.nodes[u]["TimeInfected"] += 1
                else:
                    G.nodes[u]["state"] = 2  # "recovered"
                    r[t] += 1
                    inf[t] += -1
        # check contagion
        for u in G.nodes:
            # if susceptible
            if G.nodes[u]["state"] == 0:
                nb_friend_infected = [G.nodes[n]["state"] == 1 for n in G.nodes[u]["noeux_associes"]].count(True)
                # print(nb_friend_infected)
                for n in G.nodes[u]["noeux_associes"]:
                    if G.nodes[n]["state"] == 1:  # if friend is infected
                        # with HM infect
                        if random.uniform(0,1) < G[u][n]['weight']:
                            G.nodes[u]["state"] = 1
                            inf[t] += 1
                            s[t] += -1
                            break

    return s, inf, r

def SIR_quarantine(G, Nb_inf_init, Gamma, Q, N, T):
    """ function that runs a simulation of an SIR model on a network.
    Args:
        G(nx.Graph): Graph of agents
        Gamma(int): recovery rate
        Q(int): quarintine rate
        Nb_inf_init(int): initial number of infected individuals
        N(int): number of agents (nodes)
        T(int): number of time steps simulated
    """
    A = nx.to_numpy_array(G)
    # setting initial conditions
    s = np.zeros(T)
    inf = np.zeros(T)
    r = np.zeros(T)
    inf[0] = Nb_inf_init
    s[0] = N - Nb_inf_init
    """Make a graph with some infected nodes."""
    for u in G.nodes():
        G.nodes[u]["state"] = 0
        G.nodes[u]["TimeInfected"] = 0
        G.nodes[u]["noeux_associes"] = [n for n in G.neighbors(u)]
        G.nodes[u]["quarantine"] = 0

    init = random.sample(list(G.nodes()), Nb_inf_init)
    for u in init:
        G.nodes[u]["state"] = 1
        G.nodes[u]["TimeInfected"] = 1

    # running simulation
    for t in range(1, T):
        s[t] = s[t - 1]
        inf[t] = inf[t - 1]
        r[t] = r[t - 1]
        # Check which persons have recovered
        for u in G.nodes:
            # if infected
            if G.nodes[u]["state"] == 1:
                if G.nodes[u]["TimeInfected"] < round(np.random.normal(Gamma,3,1)[0]):
                    G.nodes[u]["TimeInfected"] += 1

                else:
                    G.nodes[u]["state"] = 2  # "recovered"
                    G.nodes[u]["quarantine"] = 0
                    r[t] += 1
                    inf[t] += -1

                if G.nodes[u]["TimeInfected"] >= round(np.random.normal(Q,1.5,1)[0]):
                    G.nodes[u]["quarantine"] = 1
        # check contagion
        for u in G.nodes:
            # if susceptible
            if G.nodes[u]["state"] == 0:
                nb_friend_infected = [G.nodes[n]["state"] == 1 for n in G.nodes[u]["noeux_associes"]].count(True)
                # print(nb_friend_infected)
                for n in G.nodes[u]["noeux_associes"]:
                    if G.nodes[n]["state"] == 1 and G.nodes[n]["quarantine"] == 0:  # if friend is infected
                        # with HM infect
                        if random.uniform(0,1) < G[u][n]['weight']:
                            G.nodes[u]["state"] = 1
                            inf[t] += 1
                            s[t] += -1
                            break

    return s, inf, r


def plot_total_cases(G, T, Gamma, Q, Nb_inf_init):


    s_erdos, inf_erdos, r_erdos = SIR(G, Nb_inf_init, Gamma, N, T)
    plt.plot((100 / N) * inf_erdos, color='r', marker='o', label="Infected")
    plt.plot((100 / N) * r_erdos, color='g', marker='o', label="Recovered")

    s_erdos, inf_erdos, r_erdos = SIR_norm(G, Nb_inf_init, Gamma, N, T)
    plt.plot((100 / N) * inf_erdos, color='r', marker='$n$', label="Infected_norm")
    plt.plot((100 / N) * r_erdos, color='g', marker='$n$', label="Recovered_norm")

    s_erdos, inf_erdos, r_erdos = SIR_quarantine(G, Nb_inf_init, Gamma, Q, N, T)
    plt.plot((100 / N) * inf_erdos, color='r', marker='+', label="Infected_q")
    plt.plot((100 / N) * r_erdos, color='g', marker='+', label="Recovered_q")

    plt.xlabel("time")
    plt.ylabel("Percentage of population infected")
    plt.legend()
    plt.show()

def plot_new_cases(G, T, Gamma, Q, Nb_inf_init):
    s_erdos, inf_erdos, nb_inf_t = SIR_norm(G, Nb_inf_init, Gamma, N, T)
    plt.plot(nb_inf_t, "r", marker='o', label="Infected")

    s_erdos, inf_erdos, nb_inf_t = SIR_quarantine(G, Nb_inf_init, Gamma, Q, N, T)
    plt.plot(nb_inf_t, "r", marker='+', label="Infected_q")
    plt.xlabel("time")
    plt.ylabel("Number of new cases")

    plt.legend()
    plt.show()

if __name__ == "__main__":
    G = nx.read_edgelist('Graph.gz')
    #G = network_generator(200)
    # nx.write_edgelist(G, 'Graph.gz')
    T = 100
    # number of agents
    N = G.number_of_nodes()
    Gamma = 14
    Q = 5
    Nb_inf_init = 3

    plot_total_cases(G, T, Gamma, Q, Nb_inf_init)

    #plot_new_cases(G, T, Gamma, Q, Nb_inf_init)