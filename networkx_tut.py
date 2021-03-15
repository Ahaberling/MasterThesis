import networkx as nx

G = nx.Graph()
G.add_node(1)
G.add_nodes_from(range(1,5), color ='green')

print(G.nodes)
print(G.nodes[2])
print(G.nodes[2]['color'])
print(G.nodes.data())

G.add_edge(1,2)
G.add_edge(6,7)

print(G.edges)
print(G.nodes)