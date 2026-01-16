import torch
from torch_geometric.data import Data, Dataset
from util import average_pool_plv
import numpy as np

class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        # feature랑 tensor랑 같이 추출
        return self.X[idx], torch.tensor(self.y[idx].astype(int), dtype=torch.long) # EEG data, label


class BrainGraphDataset(Dataset):
    def __init__(self, plv_data, num_samples):
        super(BrainGraphDataset, self).__init__()
        self.plv_data = torch.tensor(plv_data, dtype=torch.float32)
        #self.plv_data = torch.from_numpy(plv_data.astype(np.float32))

        self.plv_data = average_pool_plv(
            self.plv_data
        )

        self.time_points = self.plv_data.shape[-1]
        self.num_samples = num_samples

        nodes = torch.arange(plv_data.shape[1])

        # Fully Connected 패키지라고 생각하면 됨 (https://pytorch.org/docs/stable/generated/torch.cartesian_prod.html)
        self.edge_index = torch.cartesian_prod(nodes, nodes).t()

    def len(self):
        return self.num_samples * self.time_points

    def get(self, idx):
        sample_idx = idx // self.time_points
        time_idx = idx % self.time_points

        current_plv = self.plv_data[sample_idx, :, :, time_idx]
        edge_attr = current_plv.reshape(-1, 1)

        return Data(
            edge_index=self.edge_index,
            edge_attr=edge_attr,
            num_nodes=63
        )


class CombinedDataset(Dataset):
    def __init__(self, eeg_data: EEGDataset, graph_dataset: BrainGraphDataset):
        super().__init__()

        self.eeg_data = eeg_data  # (4175, 63, 1875, 5)
        self.graph_dataset = graph_dataset
        self.time_points = graph_dataset.time_points
        self.graph_data = torch.utils.data.get_worker_info().dataset.graph_data if torch.utils.data.get_worker_info() else [graph_dataset.get(i) for i in range(len(graph_dataset))]

    def __len__(self):
        return len(self.eeg_data)  # 4175

    def __getitem__(self, idx):
        eeg_data, label = self.eeg_data[idx]  # (63, 1875, 5)

        # idx에 해당하는 973개의 그래프 데이터 가져오기
        start_idx = idx * self.time_points

        graphs = self.graph_data[start_idx: start_idx + self.time_points]

        return eeg_data.float(), label.long(), graphs