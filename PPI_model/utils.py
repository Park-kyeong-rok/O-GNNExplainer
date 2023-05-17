from torch_geometric.data import Data
from torch_geometric.utils import remove_isolated_nodes

def get_task(idx):
    def transform(data):
        return Data(x=data.x, edge_index=data.edge_index, y=data.y[:, idx])

    return transform


def get_task_rm_iso(idx):
    def transform(data):
        edge_index, _, mask = remove_isolated_nodes(data.edge_index, num_nodes=data.x.shape[0])
        return Data(x=data.x[mask], edge_index=edge_index, y=data.y[mask, idx])

    return transform


