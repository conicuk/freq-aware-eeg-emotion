import torch
import torch.nn as nn
import math
from torch_geometric.data import Batch
import torch_geometric.nn as gnn


class EncoderBlock(nn.Module):
    def __init__(self, seq_length, feature_size, in_channels=63, out_channels=63, kernel_size=(1, 3), stride=(1, 2), padding=0):
        super(EncoderBlock, self).__init__()

        self.feature_size = feature_size

        # Conv2D layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)

        # Positional Embedding
        self.positional_embedding = nn.Embedding(seq_length, out_channels)

        seq_len1 = self.conv2d_output_size(seq_length, kernel_size[1], stride[1], padding)
        seq_len2 = self.conv2d_output_size(seq_len1, kernel_size[1], stride[1], padding)
        seq_len3 = self.conv2d_output_size(seq_len2, kernel_size[1], stride[1], padding)

        # Average Pooling for Positional Embedding
        self.avg_pool1 = nn.AdaptiveAvgPool2d((1, seq_len2))
        self.avg_pool2 = nn.AdaptiveAvgPool2d((1, seq_len3))

        # Cross Attention module
        self.cross_attention = nn.MultiheadAttention(embed_dim=out_channels * feature_size, num_heads=3,
                                                     batch_first=True)

    def forward(self, x):
        device = x.device # 배치 연산 최적화
        batch_size, in_channels, seq_length, feature_size = x.shape

        # 한 번의 permute로 처리
        x = x.permute(0, 1, 3, 2).contiguous()

        # Position indices 미리 계산
        seq_indices = torch.arange(x.shape[-1], device=device)

        # Conv 연산
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        # Positional embedding 한번에 계산
        pos1 = self.positional_embedding(seq_indices[:x1.shape[-1]])
        pos1 = pos1.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, 1, -1).permute(0, 3, 2, 1)

        # In-place 덧셈 사용
        x1 = x1 + pos1
        x2 = x2 + (self.avg_pool1(pos1))
        x3 = x3 + (self.avg_pool2(self.avg_pool1(pos1)))

        # Reshape 최적화
        x2_flat = x2.transpose(1, 3).reshape(batch_size, -1, in_channels * feature_size)
        x3_flat = x3.transpose(1, 3).reshape(batch_size, -1, in_channels * feature_size)
        # Attention 계산
        x2_att = self.cross_attention(x2_flat, x3_flat, x3_flat)[0]
        x1_flat = x1.transpose(1, 3).reshape(batch_size, -1, in_channels * feature_size)
        x1_att = self.cross_attention(x1_flat, x2_att, x2_att)[0]

        return x1_att.reshape(batch_size, in_channels, -1, feature_size)

    @staticmethod
    def conv2d_output_size(length, kernel_size, stride, padding=0, dilation=1):
        return math.floor((length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)


class SpatialTemporalGNN(nn.Module):
    def __init__(self, gnn_in_channels, gat_out_channels, gru_hidden_size, num_classes=3, num_gru_layers=1, edge_dim=1,
                 k_timestamps=5):
        super(SpatialTemporalGNN, self).__init__()

        self.seq_length = None
        self.k_timestamps = k_timestamps
        self.num_classes = num_classes

        # Spatial -> GAT
        self.spatial_layer = gnn.GATv2Conv(gnn_in_channels, gat_out_channels, edge_dim=edge_dim)

        # Graph-level pooling
        self.graph_pool = gnn.MeanAggregation()

        # Temporal -> GRU
        self.temporal_layer = nn.GRU(gat_out_channels, gru_hidden_size,
                                   num_layers=num_gru_layers, batch_first=True)

        # MLP for emotion classification
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_size, gru_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(gru_hidden_size // 2, num_classes)
        )

    def forward(self, batch):
        batch_size = batch.num_graphs // self.seq_length

        # GAT 연산 최적화
        x_spatial = self.spatial_layer(batch.x, batch.edge_index, batch.edge_attr)

        # Graph pooling
        x_graph = self.graph_pool(x_spatial, ptr=batch.ptr)

        # 효율적인 reshape
        x_temporal = x_graph.view(batch_size, self.seq_length, -1).contiguous()

        # GRU + Classification
        gru_out = self.temporal_layer(x_temporal)[0]
        timestamp_scores = self.classifier(gru_out)

        # Top-k 계산 최적화
        topk_score, topk_idx = torch.topk(timestamp_scores,
                                          k=self.k_timestamps,
                                          dim=1,
                                          sorted=False)  # sorted=False로 정렬 비용 절감

        return torch.mean(topk_score, dim=1), topk_idx

    def predict_emotion(self, final_scores):
        return torch.argmax(final_scores, dim=1)


class SpatialTemporalGNN_Hidden(nn.Module):
    def __init__(self, gnn_in_channels, gat_out_channels, gru_hidden_size, num_classes=3, num_gru_layers=1, edge_dim=1):
        super(SpatialTemporalGNN_Hidden, self).__init__()

        self.seq_length = None
        self.num_classes = num_classes

        # Spatial -> GAT
        self.spatial_layer = gnn.GATv2Conv(gnn_in_channels, gat_out_channels, edge_dim=edge_dim)

        # Graph-level pooling
        self.graph_pool = gnn.MeanAggregation()

        # Temporal -> GRU
        self.temporal_layer = nn.GRU(gat_out_channels, gru_hidden_size,
                                     num_layers=num_gru_layers, batch_first=True)

        # MLP for emotion classification
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_size, gru_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(gru_hidden_size // 2, num_classes)
        )

    def forward(self, batch):
        batch_size = batch.num_graphs // self.seq_length

        # GAT 연산
        x_spatial = self.spatial_layer(batch.x, batch.edge_index, batch.edge_attr)

        # Graph pooling
        x_graph = self.graph_pool(x_spatial, ptr=batch.ptr)

        # Reshape for GRU
        x_temporal = x_graph.view(batch_size, self.seq_length, -1).contiguous()

        # GRU - 이제 output과 hidden state 모두 가져옴
        _, h_n = self.temporal_layer(x_temporal)  # h_n shape: (num_layers, batch_size, hidden_size)

        # final hidden state 사용 (마지막 layer의 hidden state)
        final_hidden = h_n[-1]  # shape: (batch_size, hidden_size)

        # Classification using final hidden state
        output = self.classifier(final_hidden)

        return output, None  # None을 반환하는 이유는 기존 코드와의 호환성을 위해

    def predict_emotion(self, final_scores):
        return torch.argmax(final_scores, dim=1)

def custom_collate_fn(batch):
    eeg_data = []
    graph_data = []
    labels = []

    for eeg, label, graph in batch:
        eeg_data.append(eeg)
        graph_data.extend(graph)
        labels.append(label)

    eeg_data = torch.stack(eeg_data).float()  # Ensure float32
    graph_data = Batch.from_data_list(graph_data)
    labels = torch.stack(labels).long()  # Ensure long for labels

    return eeg_data.float(), graph_data, labels


class En_STGNN(nn.Module):
    def __init__(self, seq_length, feature_size, in_channels
                 , out_channels, gnn_in_channels, gat_out_channels, gru_hidden_size, num_classes, k_timestamps):
        super().__init__()

        # Encoder
        self.encoder = EncoderBlock(
            seq_length=seq_length,
            feature_size=feature_size,
            in_channels=in_channels,
            out_channels=out_channels
        )

        # ST-GNN
        self.st_gnn = SpatialTemporalGNN(
            gnn_in_channels=gnn_in_channels,
            gat_out_channels=gat_out_channels,
            gru_hidden_size=gru_hidden_size,
            num_classes=num_classes,
            k_timestamps=k_timestamps
        )

        # seq_length 저장
        self.seq_length = seq_length

    def forward(self, x, graph):
        node_feature = self.encoder(x)  # (batch, n_node, seq_len, feature)

        batch, n_node, seq_len, feature = node_feature.shape

        graph.x = node_feature.reshape(batch * seq_len * n_node, -1)

        # ST-GNN에 seq_length 정보 전달
        self.st_gnn.seq_length = seq_len

        scores, indices = self.st_gnn(graph)

        return scores, indices
