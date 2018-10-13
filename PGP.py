import networkx as nx
import matplotlib
matplotlib.use("TkAgg") #Hack to ensure it works in virtual env.
import matplotlib.pyplot as plt
import numpy as np

def average_degree(G):
    return 2 * G.number_of_edges() / G.number_of_nodes()

A = np.loadtxt('Data/arenas-pgp/out.arenas-pgp', dtype=int, usecols=range(2), comments="%")
print(A)
G=nx.Graph()
for n in A:
    G.add_edge(n[0], n[1])

avg_deg = average_degree(G)
number_of_nodes = nx.number_of_nodes(G)
number_of_edges= nx.number_of_edges(G)
print("Number of Nodes: " + str(number_of_nodes))
print("Number of Edges: " + str(number_of_edges))
print(avg_deg)

nx.draw(G)
plt.savefig("output/pgp.png")

plt.gcf().clear()

plt.figure()
plt.hist([degree for node, degree in G.degree()], bins=range(0,500, 1))
plt.ylabel('Number of Nodes')
plt.xlabel('Degree')
plt.savefig("output/degreedistribution.png")