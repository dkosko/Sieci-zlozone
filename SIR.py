import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from ProjectSimulation import network_generator, G_statistics, plot_color_graph

def SIR(G, Nb_inf_init, Gamma, N, T):
    """ function that runs a simulation of an SIR model on a network.
    Args:
        Gamma(float): recovery rate
        Beta(float): infection probability
        Rho(float): initial fraction of infected individuals
        N(int): number of agents (nodes)
        T(int): number of time steps simulated
    """
    # setting initial conditions
    s = np.zeros(T)
    inf = np.zeros(T)
    r = np.zeros(T)
    inf[0] = Nb_inf_init
    s[0] = N - Nb_inf_init
    color_map = ['lime'] * N
    """Make a graph with some infected nodes."""
    for u in G.nodes():
        G.nodes[u]["state"] = 0
        G.nodes[u]["TimeInfected"] = 0
        G.nodes[u]["noeux_associes"] = [n for n in G.neighbors(u)]

    init = random.sample(list(G.nodes()), Nb_inf_init)
    for u in init:
        G.nodes[u]["state"] = 1
        G.nodes[u]["TimeInfected"] = 1
        color_map[int(u)] = 'red'
    # running simulation
    for t in range(1, T):
        #plot_color_graph(G, color_map)
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
                    color_map[int(u)] = 'lime'
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
                            color_map[int(u)] = 'red'
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
                if G.nodes[u]["TimeInfected"] < np.random.normal(Gamma,3,1)[0]:  #random.uniform(Gamma/2, Gamma*2):
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
        Nb_inf_init(int): initial number of infected individuals
        Gamma(int): recovery rate
        Q(int): quarintine rate
        N(int): number of agents (nodes)
        T(int): number of time steps simulated
    """
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

                        if random.uniform(0,1) < G[u][n]['weight']:
                            G.nodes[u]["state"] = 1
                            inf[t] += 1
                            s[t] += -1
                            break

    return s, inf, r


def plot_total_cases(G, T, Gamma, Q, Nb_inf_init):


    s, inf, r = SIR(G, Nb_inf_init, Gamma, N, T)
    plt.plot((100 / N) * inf, color='r', marker='o', label="Infected")
    plt.plot((100 / N) * r, color='g', marker='o', label="Recovered")

    s, inf, r = SIR_norm(G, Nb_inf_init, Gamma, N, T)
    new_inf = count_new_cases(s, Nb_inf_init)
    plt.plot((100 / N) * inf, color='r', marker='$n$', label="Infected_norm")
    plt.plot((100 / N) * r, color='g', marker='$n$', label="Recovered_norm")

    s, inf, r = SIR_quarantine(G, Nb_inf_init, Gamma, Q, N, T)
    new_inf_q = count_new_cases(s, Nb_inf_init)
    plt.plot((100 / N) * inf, color='r', marker='+', label="Infected_q")
    plt.plot((100 / N) * r, color='g', marker='+', label="Recovered_q")

    plt.xlabel("time")
    plt.ylabel("Percentage of population infected")
    plt.legend()
    plt.show()


    plt.plot(new_inf, "r", marker='o', label="Infected")
    plt.plot(new_inf_q, "blue", marker='.', label="Infected_q")
    plt.xlabel("time")
    plt.ylabel("Number of new cases")
    plt.title("New cases of Covid19 per day")
    plt.legend()
    plt.show()

def count_new_cases(s, Nb_inf_init):
    new_inf = [Nb_inf_init]
    T = len(s)
    for i in range(1, T):
        new_inf.append(s[i-1]-s[i])
    return new_inf


def plot_new_cases(G, T, Gamma, Q, Nb_inf_init):
    s, inf, r = SIR_norm(G, Nb_inf_init, Gamma, N, T)
    new_inf = count_new_cases(s, Nb_inf_init)
    plt.plot(new_inf, "r", marker='o', label="Infected")

    s, inf, r = SIR_quarantine(G, Nb_inf_init, Gamma, Q, N, T)
    new_inf = count_new_cases(s, Nb_inf_init)
    plt.plot(new_inf, "blue", marker='.', label="Infected_q")
    plt.xlabel("time")
    plt.ylabel("Number of new cases")

    plt.title("New cases of Covid19 per day")
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
    Nb_inf_init = 5


    plot_total_cases(G, T, Gamma, Q, Nb_inf_init)

    # plot_new_cases(G, T, Gamma, Q, Nb_inf_init)

    #SIR(G, Nb_inf_init, Gamma, N, T)


    #wskaźniki: liczba dni trwania epidemii, liczba osób przechorowanych, maksymalna liczba osób chorych jednocześnie