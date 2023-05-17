
import torch

from torch.nn import ReLU, Linear, LeakyReLU
from torch_geometric.nn import GCNConv, BatchNorm
from torch_geometric.nn import GCNConv, GraphConv, GINConv, MLP
import torch.nn as nn
import torch.nn.functional as F


class GCN1(torch.nn.Module):

    def __init__(self, num_features, num_classes,hidden_size):
        super(GCN1, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_size)
        self.relu1 = ReLU()
        self.lin = Linear(hidden_size, num_classes,bias=True)

    def forward(self, x, edge_index, edge_weights = None):

        input_lin = self.embedding(x, edge_index, edge_weights)
        self.representation = input_lin
        final = self.lin(input_lin)
        self.score = final

        return final
    def embedding(self, x, edge_index, edge_weights = None):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1))
        out1 = self.conv1(x, edge_index, edge_weights)

          # this is not used in PGExplainer
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)
        out1 = self.relu1(out1)
        return out1

class GCN4(torch.nn.Module):

    def __init__(self, num_features, num_classes,hidden_size):
        super(GCN4, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_size)
        self.relu1 = ReLU()
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.relu2 = ReLU()
        self.conv3 = GCNConv(hidden_size, hidden_size)
        self.relu3 = ReLU()
        self.conv4 = GCNConv(hidden_size, hidden_size)
        self.relu4 = ReLU()
        self.lin = Linear(hidden_size, num_classes,bias=True)

    def forward(self, x, edge_index, edge_weights = None):

        input_lin = self.embedding(x, edge_index, edge_weights)
        self.representation = input_lin
        final = self.lin(input_lin)
        self.score = final

        return final
    def embedding(self, x, edge_index, edge_weights = None):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1))
        out1 = self.conv1(x, edge_index, edge_weights)
          # this is not used in PGExplainer
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)
        out1 = self.relu1(out1)
        out2 = self.conv2(out1, edge_index, edge_weights)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)
        out2 = self.relu2(out2)

        out3 = self.conv3(out2, edge_index, edge_weights)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1)  # this is not used in PGExplainer
        out3 = self.relu3(out3)

        out4 = self.conv4(out3, edge_index, edge_weights)
        out4 = torch.nn.functional.normalize(out4, p=2, dim=1)  # this is not used in PGExplainer
        out4 = self.relu4(out4)

        return out4


class aggg_GCN3(torch.nn.Module):

    def __init__(self, num_features, num_classes,hidden_size):
        super(agg_GCN3, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_size)
        self.relu1 = ReLU()
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.relu2 = ReLU()
        self.conv3 = GCNConv(hidden_size, hidden_size)
        self.relu3 = ReLU()
        self.lin = Linear(hidden_size*3, num_classes,bias=True)

    def forward(self, x, edge_index, edge_weights = None):

        input_lin = self.embedding(x, edge_index, edge_weights)
        self.representation = input_lin
        final = self.lin(input_lin)
        self.score = final

        return final
    def embedding(self, x, edge_index, edge_weights = None):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1))
        out1 = self.conv1(x, edge_index, edge_weights)

          # this is not used in PGExplainer
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)
        out1 = self.relu1(out1)



        out2 = self.conv2(out1, edge_index, edge_weights)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)
        out2 = self.relu2(out2)

        out3 = self.conv3(out2, edge_index, edge_weights)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1)  # this is not used in PGExplainer
        out3 = self.relu3(out3)


        return torch.concat([out1, out2, out3], dim=1)

class NodeGCN1(torch.nn.Module):
    """
    A graph clasification model for nodes decribed in https://arxiv.org/abs/2011.04573.
    This model consists of 3 stacked GCN layers and batch norm, followed by a linear layer.
    """
    def __init__(self, num_features, num_classes, hidden_size):
        super(NodeGCN1, self).__init__()
        self.embedding_size = hidden_size
        self.conv1 = GCNConv(num_features, hidden_size)
        self.relu1 = LeakyReLU()
        self.bn1 = BatchNorm(hidden_size)        # BN is not used in GNNExplainer
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.relu2 = LeakyReLU()
        self.bn2 = BatchNorm(hidden_size)
        self.conv3 = GCNConv(hidden_size, hidden_size)
        self.relu3 = LeakyReLU()
        self.lin = Linear(self.embedding_size*3, num_classes)

    def forward(self, x, edge_index, edge_weights=None):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1))
        input_lin = self.embedding(x, edge_index, edge_weights)
        out = self.lin(input_lin)
        return out

    def embedding(self, x, edge_index, edge_weights=None):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1))

        out1 = self.conv1(x, edge_index, edge_weights)
        out1 = self.relu1(out1)
        out1 = self.bn1(out1)


        out2 = self.conv2(out1, edge_index, edge_weights)
        out2 = self.relu2(out2)
        out2 = self.bn2(out2)


        out3 = self.conv3(out2, edge_index, edge_weights)
        out3 = self.relu3(out3)

        return torch.concat([out1, out2, out3], dim=1)

class GCN2(torch.nn.Module):
    """
    A graph clasification model for nodes decribed in https://arxiv.org/abs/2011.04573.
    This model consists of 3 stacked GCN layers and batch norm, followed by a linear layer.
    """

    def __init__(self, num_features, num_classes, hidden_size):
        super(GCN2, self).__init__()
        self.embedding_size = hidden_size
        #self.conv1 = GCNConv(num_features, hidden_size)

        self.conv1 = GINConv(nn.Sequential(nn.Linear(num_features, hidden_size),
                                  nn.ReLU(), nn.Linear(hidden_size,hidden_size)))
        self.relu1 = ReLU()
        self.bn1 = BatchNorm(hidden_size, track_running_stats=False)  # BN is not used in GNNExplainer
        #self.conv2 = GCNConv(hidden_size, hidden_size)
        #mlp2 = torch.nn.Linear(hidden_size, hidden_size)
        self.conv2 = GINConv(nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                  nn.ReLU(), nn.Linear(hidden_size,hidden_size)))
        self.relu2 = ReLU()
        self.lin = Linear(self.embedding_size, num_classes)

    def forward(self, x, edge_index, edge_weights=None):
        input_lin = self.embedding(x, edge_index, edge_weights)
        self.representation = input_lin

        out = self.lin(input_lin)
        self.score = out
        return out

    def embedding(self, x, edge_index, edge_weights=None):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1))

        #out1 = self.conv1(x, edge_index, edge_weights)
        out1 = self.conv1(x, edge_index)

        out1 = self.bn1(out1)
        #out1 = self.bn4(out1)

        #out2 = self.conv2(out1, edge_index, edge_weights)
        out2 = self.conv2(out1, edge_index)
        print(out2.shape)


        return out2

class GCN3(torch.nn.Module):
    """
    A graph clasification model for nodes decribed in https://arxiv.org/abs/2011.04573.
    This model consists of 3 stacked GCN layers and batch norm, followed by a linear layer.
    """

    def __init__(self, num_features, num_classes, hidden_size):
        super(GCN3, self).__init__()
        self.embedding_size = hidden_size
        self.conv1 = GCNConv(num_features, hidden_size)
        self.relu1 = ReLU()
        self.bn1 = BatchNorm(hidden_size, track_running_stats=True, momentum=1.0)  # BN is not used in GNNExplainer
        self.bn2 = BatchNorm(hidden_size, track_running_stats=True, momentum=1.0)  # BN is not used in GNNExplainer
        self.bn3 = BatchNorm(hidden_size, track_running_stats=True, momentum=1.0)  # BN is not used in GNNExplainer
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.relu2 = ReLU()
        self.conv3 = GCNConv(hidden_size, hidden_size)
        self.relu3 = ReLU()
        self.lin = Linear(self.embedding_size, num_classes)

    def forward(self, x, edge_index, edge_weights=None):
        input_lin = self.embedding(x, edge_index, edge_weights)
        self.representation = input_lin

        out = self.lin(input_lin)
        self.score = out
        return out

    def embedding(self, x, edge_index, edge_weights=None):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1)).to(torch.device(x.device))

        out1 = self.conv1(x, edge_index, edge_weights)
        out1 = self.relu1(out1)
        out1 = self.bn1(out1)

        out2 = self.conv2(out1, edge_index, edge_weights)
        out2 = self.relu2(out2)
        out2 = self.bn2(out2)

        out3 = self.conv3(out2, edge_index, edge_weights)
        out3 = self.relu3(out3)
        out3 = self.bn3(out3)
        return out3

class GCN3_encoder(torch.nn.Module):
    """
    A graph clasification model for nodes decribed in https://arxiv.org/abs/2011.04573.
    This model consists of 3 stacked GCN layers and batch norm, followed by a linear layer.
    """

    def __init__(self, num_features, hidden_size):
        super(GCN3_encoder, self).__init__()
        self.embedding_size = hidden_size
        self.conv1 = GCNConv(num_features, hidden_size)
        self.relu1 = ReLU()
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.relu2 = ReLU()
        self.conv3 = GCNConv(hidden_size, hidden_size)
        self.relu3 = ReLU()
        self.bn1 = BatchNorm(hidden_size, track_running_stats=True, momentum=1.0)  # BN is not used in GNNExplainer
        self.bn2 = BatchNorm(hidden_size, track_running_stats=True, momentum=1.0)  # BN is not used in GNNExplainer
        self.bn3 = BatchNorm(hidden_size, track_running_stats=True, momentum=1.0)  # BN is not used in GNNExplainer

    def forward(self, x, edge_index, edge_weights=None):

        input_lin = self.embedding(x, edge_index, edge_weights)

        return input_lin

    def embedding(self, x, edge_index, edge_weights=None):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1)).to(torch.device(x.device))

        out1 = self.conv1(x, edge_index, edge_weights)
        out1 = self.relu1(out1)
        out1 = self.bn1(out1)

        out2 = self.conv2(out1, edge_index, edge_weights)
        out2 = self.relu2(out2)
        out2 = self.bn2(out2)

        out3 = self.conv3(out2, edge_index, edge_weights)
        out3 = self.relu3(out3)
        out3 = self.bn3(out3)
        return out3



class GCN3_classifier(torch.nn.Module):
    """
    A graph clasification model for nodes decribed in https://arxiv.org/abs/2011.04573.
    This model consists of 3 stacked GCN layers and batch norm, followed by a linear layer.
    """

    def __init__(self,num_classes, hidden_size):
        super(GCN3_classifier, self).__init__()
        self.embedding_size = hidden_size

        self.lin = Linear(self.embedding_size, num_classes)

    def forward(self, x):

        return self.lin(x)





class agg_NodeGCN1(torch.nn.Module):
    """
    A graph clasification model for nodes decribed in https://arxiv.org/abs/2011.04573.
    This model consists of 3 stacked GCN layers and batch norm, followed by a linear layer.
    """
    def __init__(self, num_features, num_classes, hidden_size):
        super(agg_NodeGCN1, self).__init__()
        self.embedding_size = hidden_size*3
        self.conv1 = GCNConv(num_features, hidden_size)
        self.relu1 = ReLU()
        self.bn1 = BatchNorm(hidden_size, track_running_stats=False)        # BN is not used in GNNExplainer
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.relu2 = ReLU()
        self.bn2 = BatchNorm(hidden_size, track_running_stats=False)
        self.conv3 = GCNConv(hidden_size, hidden_size)
        self.relu3 = ReLU()
        self.bn3 = BatchNorm(hidden_size, track_running_stats=False)
        self.lin = Linear(self.embedding_size, num_classes)

    def forward(self, x, edge_index, edge_weights=None):
        input_lin = self.embedding(x, edge_index, edge_weights)
        out = self.lin(input_lin)
        return out

    def embedding(self, x, edge_index, edge_weights=None):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1))
        stack = []
        out1 = self.conv1(x, edge_index, edge_weights)
        out1 = self.bn1(out1)
        out1 = self.relu1(out1)
        #out1 = self.bn1(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index, edge_weights)
        out2 = self.bn2(out2)
        out2 = self.relu2(out2)
        #out2 = self.bn2(out2)
        stack.append(out2)

        out3 = self.conv3(out2, edge_index, edge_weights)
        out3 = self.bn3(out3)
        out3 = self.relu3(out3)
        #out3 = self.bn3(out3)
        stack.append(out3)

        input_lin = torch.cat(stack, dim=1)
        return input_lin

# class Net(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels):
#         super().__init__()
#
#         out_channels = 1
#         self.conv1 = GraphConv(in_channels, hidden_channels)
#         self.conv2 = GraphConv(hidden_channels, hidden_channels)
#         self.conv3 = GraphConv(hidden_channels, hidden_channels)
#         self.lin = torch.nn.Linear(3 * hidden_channels, out_channels)
#
#     def embedding(self, x0, edge_index, edge_weights=None):
#         x1 = F.relu(self.conv1(x0, edge_index, edge_weights))
#         x2 = F.relu(self.conv2(x1, edge_index, edge_weights))
#         x3 = F.relu(self.conv3(x2, edge_index, edge_weights))
#         x = torch.cat([x1, x2, x3], dim=-1)
#         return x
#
#
#     def forward(self, x0, edge_index, edge_weights=None):
#         input_lin = self.embedding(x0, edge_index, edge_weights)
#         self.representation = input_lin
#         out = self.lin(input_lin)
#         self.score = out
#         return out


# class Net(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels):
#         super().__init__()
#
#         out_channels = 1
#         self.conv1 = GraphConv(in_channels, hidden_channels)
#         self.conv2 = GraphConv(hidden_channels, hidden_channels)
#         self.conv3 = GraphConv(hidden_channels, hidden_channels)
#         self.lin = torch.nn.Linear(3 * hidden_channels, out_channels)
#
#     def embedding(self, x0, edge_index, edge_weights=None):
#         x1 = F.relu(self.conv1(x0, edge_index, edge_weights))
#         x2 = F.relu(self.conv2(x1, edge_index, edge_weights))
#         x3 = F.relu(self.conv3(x2, edge_index, edge_weights))
#         x = torch.cat([x1, x2, x3], dim=-1)
#         return x
#
#
#     def forward(self, x0, edge_index, edge_weights=None):
#         input_lin = self.embedding(x0, edge_index, edge_weights)
#         self.representation = input_lin
#         out = self.lin(input_lin)
#         self.score = out
#         return out
#
# class Netsimple(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels):
#         super().__init__()
#         out_channels = 1
#         self.conv1 = GraphConv(in_channels, hidden_channels)
#         self.conv2 = GraphConv(hidden_channels, hidden_channels)
#         self.conv3 = GraphConv(hidden_channels, hidden_channels)
#         self.lin = torch.nn.Linear(10, 1)
#     def embedding(self, x0, edge_index, edge_weights=None):
#         x1 = F.relu(self.conv1(x0, edge_index, edge_weights))
#         x2 = F.relu(self.conv2(x1, edge_index, edge_weights))
#         x3 = F.relu(self.conv3(x2, edge_index, edge_weights))
#         return x3
#
#
#     def forward(self, x0, edge_index, edge_weights=None):
#         input_lin = self.embedding(x0, edge_index, edge_weights)
#         self.representation = input_lin
#         out = self.lin(input_lin)
#         self.score = out
#         return out

class Netsimple2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        out_channels = 1
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(10, 1)
    def embedding(self, x0, edge_index, edge_weights=None):
        x1 = F.relu(self.conv1(x0, edge_index, edge_weights))
        x2 = F.relu(self.conv2(x1, edge_index, edge_weights))
        return x2


    def forward(self, x0, edge_index, edge_weights=None):
        input_lin = self.embedding(x0, edge_index, edge_weights)
        self.representation = input_lin
        out = self.lin(input_lin)
        self.score = out
        return out

class attention(torch.nn.Module):
    def __init__(self, embed_dim):
        super(attention,self).__init__()
        self.embed_dim = embed_dim

        self.weight = torch.nn.Parameter(torch.empty(embed_dim, embed_dim))
        #torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.kaiming_uniform_(self.weight)
        self.register_parameter('weight', self.weight)
    def forward(self, query, key):

        x = torch.matmul(key, self.weight)

        #print(f'wegith: {self.weight}')
        score = torch.matmul(x, query)
        #score = torch.unsqueeze(torch.softmax(score/self.embed_dim**(1/2), dim= 0), dim=1)
        score = torch.unsqueeze(torch.softmax(score /key.shape[0], dim=0), dim=1)

        self.score = score
        #print(score)

        out = torch.mul(key, score)
        # print(out.shape)
        # exit()
        out = torch.sum(out, dim= 0)
        return out

class agg_GCN3(torch.nn.Module):
    """
    A graph clasification model for nodes decribed in https://arxiv.org/abs/2011.04573.
    This model consists of 3 stacked GCN layers and batch norm, followed by a linear layer.
    """

    def __init__(self, num_features, num_classes, hidden_size):
        super(agg_GCN3, self).__init__()
        self.embedding_size = hidden_size
        self.conv1 = GCNConv(num_features, hidden_size)
        torch.nn.init.xavier_uniform_(self.conv1.lin.weight)
        self.relu1 = ReLU()
        #self.bn1 = BatchNorm(hidden_size, track_running_stats=True)  # BN is not used in GNNExplainer
        self.conv2 = GCNConv(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.conv2.lin.weight)
        self.relu2 = ReLU()
        #self.bn2 = BatchNorm(hidden_size, track_running_stats=True)
        self.conv3 = GCNConv(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.conv3.lin.weight)
        self.relu3 = ReLU()
        #self.bn3 = BatchNorm(hidden_size, track_running_stats=True)
        self.lin = Linear(self.embedding_size*3, num_classes)

    def forward(self, x, edge_index, edge_weights=None):
        input_lin = self.embedding(x, edge_index, edge_weights)
        self.representation = input_lin

        out = self.lin(input_lin)
        self.score = out
        return out

    def embedding(self, x, edge_index, edge_weights=None):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1))

        out1 = self.conv1(x, edge_index, edge_weights)
        #out1 = self.bn1(out1)
        out1 = self.relu1(out1)
        #out1 = self.bn4(out1)

        out2 = self.conv2(out1, edge_index, edge_weights)
        #out2 = self.bn2(out2)
        out2 = self.relu2(out2)
        #out2 = self.bn5(out2)

        out3 = self.conv3(out2, edge_index, edge_weights)
        #out3 = self.bn3(out3)
        out3 = self.relu3(out3)
        #out3 = self.bn6(out3)
        return torch.concat((out1, out2, out3), dim=1)

class agg_GCN3_encoder(torch.nn.Module):
    """
    A graph clasification model for nodes decribed in https://arxiv.org/abs/2011.04573.
    This model consists of 3 stacked GCN layers and batch norm, followed by a linear layer.
    """

    def __init__(self, num_features, hidden_size):
        super(agg_GCN3_encoder, self).__init__()
        self.embedding_size = hidden_size
        self.conv1 = GCNConv(num_features, hidden_size)
        self.relu1 = ReLU()
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.relu2 = ReLU()
        self.conv3 = GCNConv(hidden_size, hidden_size)
        self.relu3 = ReLU()

    def forward(self, x, edge_index, edge_weights=None):
        input_lin = self.embedding(x, edge_index, edge_weights)

        return input_lin

    def embedding(self, x, edge_index, edge_weights=None):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1))

        out1 = self.conv1(x, edge_index, edge_weights)
        out1 = self.relu1(out1)
        #out1 = self.bn4(out1)

        out2 = self.conv2(out1, edge_index, edge_weights)
        out2 = self.relu2(out2)
        #out2 = self.bn5(out2)

        out3 = self.conv3(out2, edge_index, edge_weights)
        out3 = self.relu3(out3)
        #out3 = self.bn6(out3)
        return torch.concat((out1, out2, out3), dim=1)

class agg_GCN3_classifier(torch.nn.Module):
    """
    A graph clasification model for nodes decribed in https://arxiv.org/abs/2011.04573.
    This model consists of 3 stacked GCN layers and batch norm, followed by a linear layer.
    """

    def __init__(self,num_classes, hidden_size):
        super(agg_GCN3_classifier, self).__init__()
        self.embedding_size = hidden_size

        self.lin = Linear(self.embedding_size*3, num_classes)

    def forward(self, x):

        return self.lin(x)

class Net(torch.nn.Module):
    """
    A graph clasification model for nodes decribed in https://arxiv.org/abs/2011.04573.
    This model consists of 3 stacked GCN layers and batch norm, followed by a linear layer.
    """

    def __init__(self, num_features, num_classes, hidden_size):
        super(Net, self).__init__()
        self.embedding_size = hidden_size
        self.conv1 = GCNConv(num_features, hidden_size)
        self.relu1 = ReLU()
        self.bn1 = BatchNorm(hidden_size, track_running_stats=True, momentum=1.0)  # BN is not used in GNNExplainer
        #self.bn1 = BatchNorm(hidden_size, track_running_stats=True)  # BN is not used in GNNExplainer
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.relu2 = ReLU()
        self.bn2 = BatchNorm(hidden_size, track_running_stats=True, momentum=1.0)  # BN is not used in GNNExplainer
        #self.bn2 = BatchNorm(hidden_size, track_running_stats=True)
        self.conv3 = GCNConv(hidden_size, hidden_size)
        self.bn3 = BatchNorm(hidden_size, track_running_stats=True, momentum=1.0)  # BN is not used in GNNExplainer
        self.relu3 = ReLU()
        #self.bn3 = BatchNorm(hidden_size, track_running_stats=True)
        self.lin = Linear(self.embedding_size*3, 1)

    def forward(self, x, edge_index, edge_weights=None):


        input_lin = self.embedding(x, edge_index, edge_weights)
        self.representation = input_lin

        out = self.lin(input_lin)
        self.score = out
        return out

    def embedding(self, x, edge_index, edge_weights=None):

        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1)).to(x.device)

        out1 = self.conv1(x, edge_index, edge_weights)


        #out1 = self.bn1(out1)
        out1 = self.relu1(out1)
        out1 = self.bn1(out1)

        out2 = self.conv2(out1, edge_index, edge_weights)

        #out2 = self.bn2(out2)
        out2 = self.relu2(out2)
        out2 = self.bn2(out2)

        out3 = self.conv3(out2, edge_index, edge_weights)


        #out3 = self.bn3(out3)
        out3 = self.relu3(out3)
        out3 = self.bn3(out3)
        return torch.concat([out1, out2, out3], dim=1)


class Net_classifier(torch.nn.Module):
    """
    A graph clasification model for nodes decribed in https://arxiv.org/abs/2011.04573.
    This model consists of 3 stacked GCN layers and batch norm, followed by a linear layer.
    """

    def __init__(self,num_classes, hidden_size):
        super(Net_classifier, self).__init__()
        self.embedding_size = hidden_size*3

        self.lin = Linear(self.embedding_size, 1)

    def forward(self, x):

        return self.lin(x)

class Net_encoder(torch.nn.Module):
    """
    A graph clasification model for nodes decribed in https://arxiv.org/abs/2011.04573.
    This model consists of 3 stacked GCN layers and batch norm, followed by a linear layer.
    """

    def __init__(self, num_features, hidden_size):
        super(Net_encoder, self).__init__()
        self.embedding_size = hidden_size
        self.conv1 = GCNConv(num_features, hidden_size)
        self.relu1 = ReLU()
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.relu2 = ReLU()
        self.conv3 = GCNConv(hidden_size, hidden_size)
        self.relu3 = ReLU()
        self.bn1 = BatchNorm(hidden_size, track_running_stats=True, momentum=1.0)  # BN is not used in GNNExplainer
        self.bn2 = BatchNorm(hidden_size, track_running_stats=True, momentum=1.0)  # BN is not used in GNNExplainer
        self.bn3 = BatchNorm(hidden_size, track_running_stats=True, momentum=1.0)  # BN is not used in GNNExplainer

    def forward(self,x,edge_index, edge_weights = None):

        input_lin = self.embedding(x, edge_index, edge_weights=edge_weights)

        return input_lin

    def embedding(self, x, edge_index, edge_weights=None):

        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1)).to(x.device)

        out1 = self.conv1(x, edge_index, edge_weights)
        out1 = self.relu1(out1)
        out1 = self.bn1(out1)

        out2 = self.conv2(out1, edge_index, edge_weights)
        out2 = self.relu2(out2)
        out2 = self.bn2(out2)

        out3 = self.conv3(out2, edge_index, edge_weights)
        out3 = self.relu3(out3)
        out3 = self.bn3(out3)
        return torch.concat([out1, out2, out3], dim=1)