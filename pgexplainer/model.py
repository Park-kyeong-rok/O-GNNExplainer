
import torch

from torch.nn import ReLU, Linear, LeakyReLU
from torch_geometric.nn import GCNConv, BatchNorm
from torch_geometric.nn import GCNConv, GraphConv, GINConv, MLP



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
        #self.bn1 = BatchNorm(hidden_size, track_running_stats=True)  # BN is not used in GNNExplainer
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.relu2 = ReLU()
        self.bn2 = BatchNorm(hidden_size, track_running_stats=True, momentum=1.0)  # BN is not used in GNNExplainer
        #self.bn2 = BatchNorm(hidden_size, track_running_stats=True)
        self.conv3 = GCNConv(hidden_size, hidden_size)
        self.bn3 = BatchNorm(hidden_size, track_running_stats=True, momentum=1.0)  # BN is not used in GNNExplainer
        self.relu3 = ReLU()
        #self.bn3 = BatchNorm(hidden_size, track_running_stats=True)
        self.lin = Linear(self.embedding_size, num_classes)

    def forward(self, x, edge_index, edge_weights=None):

        input_lin = self.embedding(x, edge_index, edge_weights)
        self.representation = input_lin
        out = self.lin(input_lin)
        self.score = out
        return out

    def embedding(self, x, edge_index, edge_weights=None):
        device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1)).to(device)

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
        return out3

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

    def forward(self,x, edge_index, edge_weights = None):


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