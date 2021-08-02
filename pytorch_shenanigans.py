#python -c "import torch; print(torch.__version__)"             1.9.0+cu102
#python -c "import torch; print(torch.version.cuda)"            10.2



'''
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html       # done
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html
pip install torch-geometric                                                                 # done
'''

if __name__ == '__main__':



#--- Import Libraries ---#
    print('\n#--- Import Libraries ---#\n')

    import pickle as pk
    import numpy as np

    import networkx as nx
    from cdlib import algorithms
    # import wurlitzer                   #not working for windows

    import tqdm
    import os

    import torch
    from torch_geometric.data import Data



#--- Initialization ---#
    print('\n# --- Initialization ---#\n')

    os.chdir('D:/Universitaet Mannheim/MMDS 7. Semester/Master Thesis/Outline/Data/Cleaning Robots')

    with open('windows_topicOccu', 'rb') as handle:
        topicOccu = pk.load(handle)


#--- Torch Magic ---#


    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)

    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    print(data)
    #>>> Data(edge_index=[2, 4], x=[3, 1])