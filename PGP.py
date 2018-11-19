import networkx as nx
import matplotlib
matplotlib.use("TkAgg") #Hack to ensure it works in virtual env.
import matplotlib.pyplot as plt
import numpy as np
import collections
from scipy.optimize import curve_fit
from scipy.special import factorial
from scipy import optimize

def average_degree(G):
    return 2 * G.number_of_edges() / G.number_of_nodes()

A = np.loadtxt('Data/arenas-pgp/out.arenas-pgp', dtype=int, usecols=range(2), comments="%")
print(A)
G=nx.Graph()
for n in A:
    G.add_edge(n[0], n[1])


def moment(graph_data,n):
    degree_per_node = nx.degree(graph_data)

    val = 0
    for deg in degree_per_node:
         val += deg[1] ** n

    return val / float(nx.number_of_nodes(graph_data))

nx.write_gml(G, "output/pgp.gml", stringizer=str)

avg_deg = average_degree(G)
number_of_nodes = nx.number_of_nodes(G)
number_of_edges= nx.number_of_edges(G)
print("Number of Nodes: " + str(number_of_nodes))
print("Number of Edges: " + str(number_of_edges))
print("avg degree" + str(avg_deg))
print("1st moment" + str(moment(G, 1)))
print("2nd moment" + str(moment(G, 2)))
print("3rd moment" + str(moment(G, 3)))



degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())

fig, ax = plt.subplots()
plt.bar(deg, cnt, width=0.80, color='b')
plt.title("Degree Histogram")
plt.ylabel("Count")
plt.xlabel("Degree")
plt.savefig("output/degreedistribution.png")

plt.gcf().clear()

plt.loglog(degree_sequence,'b-',marker='o')
plt.title("Degree rank plot (log scale)")
plt.ylabel("degree")
plt.xlabel("rank")
plt.savefig("output/degreedistribution-log.png")

plt.gcf().clear()


data = [d for n, d in G.degree()]
maxDegree = np.max(data)
minDegree = np.min(data)

# the bins should be of integer width, because poisson is an integer distribution
entries, bin_edges, patches = plt.hist(data, bins=11, range=[-0.5, 50])

# poisson function, parameter lamb is the fit parameter
def poisson(k, lamb):
    return (lamb**k/factorial(k)) * np.exp(-lamb)

# calculate binmiddles
bin_middles = 0.5*(bin_edges[1:] + bin_edges[:-1])

# fit with curve_fit
parameters, cov_matrix = curve_fit(poisson, bin_middles, entries) 

# plot poisson-deviation with fitted parameter
x_plot = np.linspace(minDegree, maxDegree, len(data))

plt.plot(x_plot, poisson(x_plot, *parameters), 'r-')
plt.savefig("output/degreedistribution-poissonfit.png")

