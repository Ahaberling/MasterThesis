if __name__ == '__main__':

    #--- Import Libraries ---#
    print('\n#--- Import Libraries ---#\n')

    # Utilities
    import random as rd

    # Data Handling
    import numpy as np
    import pickle as pk

    # Pytorch
    import torch
    import torch_geometric.data as data
    from torch_geometric.nn import Node2Vec

    # Visualization
    import matplotlib.pyplot as plt

    # Model
    from sklearn.model_selection import cross_val_score
    from xgboost import XGBRegressor


    #--- Initialization ---#
    print('\n#--- Initialization ---#\n')

    # directory
    path = 'D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/new'

    # --- Import Data ---#
    with open('topicProject_graphs', 'rb') as handle:
        topicNetwork = pk.load(handle)


    #########################################################################
    # --- Extract and transform the edges and their weight of window 4500 ---#
    #########################################################################

    edges_4500 = []

    # extract all edges
    for edge in topicNetwork['window_4500'].edges(data=True):
        edge_container = [int(edge[0][6:9]), int(edge[1][6:9]), edge[2]['weight']]
        edges_4500.append(edge_container)

    edges_4500 = np.array(edges_4500)

    # Sort edges
    edges_4500 = edges_4500[edges_4500[:, 1].argsort()]
    edges_4500 = edges_4500[edges_4500[:, 0].argsort(kind='mergesort')]

    # Split edges from weights:
    # Split into edges
    edge_ids_4500 = edges_4500[:, 0:2].astype(int).T

    # Split into weights
    edge_weights_4500 = edges_4500[:, 2:3].T[0]

    # Make edges undirectional
    edge_pos1 = np.append(edge_ids_4500[0], edge_ids_4500[1])
    edge_pos2 = np.append(edge_ids_4500[1], edge_ids_4500[0])
    edge_ids_4500 = torch.tensor([edge_pos1, edge_pos2])

    # Make weights undirectional
    edge_weights_4500 = np.append(edge_weights_4500, edge_weights_4500)

    #########################################################################
    # --- Extract and transform the edges and their weight of window 4860 ---#
    #########################################################################
    edges_4860 = []

    # extract all edges
    for edge in topicNetwork['window_4860'].edges(data=True):
        edge_container = [int(edge[0][6:9]), int(edge[1][6:9]), edge[2]['weight']]
        edges_4860.append(edge_container)

    edges_4860 = np.array(edges_4860)

    # Sort edges
    edges_4860 = edges_4860[edges_4860[:, 1].argsort()]
    edges_4860 = edges_4860[edges_4860[:, 0].argsort(kind='mergesort')]

    # Split edges from weights:
    # Split into edges
    edge_ids_4860 = edges_4860[:, 0:2].astype(int).T

    # Split into weights
    edge_weights_4860 = edges_4860[:, 2:3].T[0]

    ############################
    # --- Create fake edges ---#
    ############################

    rd.seed(0)

    i = 0
    while i < (np.shape(edges_4860)[0] * 2):
        random_node1 = rd.randint(0, 329)
        random_node2 = rd.randint(0, 329)

        if random_node1 != random_node2:
            random_edge = [random_node1, random_node2]
            appending = True

            for j in edge_ids_4860.T:
                if random_edge == list(j):
                    appending = False

            if appending == True:
                random_edge_transf = np.array([[random_edge[0]], [random_edge[1]]])
                edge_ids_4860 = np.hstack([edge_ids_4860, np.array([[random_node1], [random_node2]])])

                edge_weights_4860 = np.hstack([edge_weights_4860, 0])
                i = i + 1

    edge_ids_4860 = torch.tensor([edge_ids_4860[0], edge_ids_4860[1]])

    graph_4500 = data.Data(edge_index=edge_ids_4500, edge_attr=edge_weights_4500)
    graph_4860 = data.Data(edge_index=edge_ids_4860, edge_attr=edge_weights_4860)



    #--- specify node2vec embedding model and optimized ---#

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Node2Vec(graph_4500.edge_index, embedding_dim=128, walk_length=20, context_size=10, walks_per_node=10,
                     num_negative_samples=1, p=1, q=1, sparse=True).to(device)

    loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)


    #--- train node2vec embedding model ---#

    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)


    plot_x = []
    plot_y = []
    for epoch in range(1, 301):
        loss = train()
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
            plot_x.append(epoch)
            plot_y.append(loss)


    #--- Visualization Loss ---#
    fig, ax = plt.subplots()

    ax.plot(plot_x, plot_y, color='darkred')
    plt.xlabel("Node2Vec Epochs")
    plt.ylabel("Node2Vec Loss")
    plt.savefig('Node2Vec_loss_300.png')
    plt.close()



    z = model()
    # from tensor to numpy
    emb_128 = z.detach().cpu().numpy()



    edge_embedding_4860 = []
    for u, v in graph_4860.edge_index.t():
        edge_embedding_4860.append(np.mean([emb_128[u], emb_128[v]], 0))


    #--- XGBoost ---X

    scores = cross_val_score(XGBRegressor(objective='reg:squarederror'), edge_embedding_4860, graph_4860.edge_attr,
                             scoring='neg_mean_squared_error', cv=10)
    print((-scores) ** 0.5)  # root mean squared error
    print(np.mean((-scores) ** 0.5))







