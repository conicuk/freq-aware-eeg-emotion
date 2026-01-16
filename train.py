import pytorch_lightning as pl
from torchmetrics import Accuracy
import torch.nn as nn
import torch
from model import EncoderBlock, SpatialTemporalGNN, SpatialTemporalGNN_Hidden


class LightningEnSTGNN_Hidden(pl.LightningModule):
    def __init__(self, seq_length, feature_size, in_channels, out_channels,
                 gnn_in_channels, gat_out_channels, gru_hidden_size,
                 num_classes,
                 #k_timestamps,
                 learning_rate, batch_size):
        super().__init__()

        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Model components
        self.encoder = EncoderBlock(
            seq_length=seq_length,
            feature_size=feature_size,
            in_channels=in_channels,
            out_channels=out_channels
        )

        self.st_gnn = SpatialTemporalGNN_Hidden(
            gnn_in_channels=gnn_in_channels,
            gat_out_channels=gat_out_channels,
            gru_hidden_size=gru_hidden_size,
            num_classes=num_classes
            #k_timestamps=k_timestamps
        )

        # Metrics
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, graph):
        node_feature = self.encoder(x)
        batch, n_node, seq_len, feature = node_feature.shape
        graph.x = node_feature.reshape(batch * seq_len * n_node, -1)
        self.st_gnn.seq_length = seq_len
        scores, indices = self.st_gnn(graph)
        return scores, indices

    def training_step(self, batch, batch_idx):
        batch_eeg, batch_graphs, batch_labels = batch
        outputs, _ = self(batch_eeg, batch_graphs)
        loss = self.criterion(outputs, batch_labels)

        # Calculate accuracy
        preds = outputs.argmax(dim=1)
        acc = self.train_accuracy(preds, batch_labels)

        # Log metrics
        self.log('train_loss/batch', loss, prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log('train_acc/batch', acc, prog_bar=True, sync_dist=True, batch_size=self.batch_size)

        return loss

    def on_train_epoch_end(self):
        # Log epoch-level training accuracy
        train_acc = self.train_accuracy.compute()
        self.log('train_acc/epoch', train_acc, prog_bar=True, sync_dist=True)
        self.train_accuracy.reset()

    def validation_step(self, batch, batch_idx):
        batch_eeg, batch_graphs, batch_labels = batch
        outputs, _ = self(batch_eeg, batch_graphs) #top-k 없앰 (outputs, indices)
        loss = self.criterion(outputs, batch_labels)

        # Calculate accuracy
        preds = outputs.argmax(dim=1)
        acc = self.val_accuracy(preds, batch_labels)

        # Log metrics
        self.log('val_loss', loss, prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log('val_acc', acc, prog_bar=True, sync_dist=True, batch_size=self.batch_size)

        return {'val_loss': loss, 'val_acc': acc}

    def on_validation_epoch_end(self):
        val_acc = self.val_accuracy.compute()
        self.log('val_acc/epoch', val_acc, prog_bar=True, sync_dist=True)
        self.val_accuracy.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.learning_rate, weight_decay=0.00001)
        return optimizer


class LightningEnSTGNN(pl.LightningModule):
    def __init__(self, seq_length, feature_size, in_channels, out_channels,
                 gnn_in_channels, gat_out_channels, gru_hidden_size,
                 num_classes,k_timestamps,learning_rate, batch_size):
        super().__init__()

        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Model components
        self.encoder = EncoderBlock(
            seq_length=seq_length,
            feature_size=feature_size,
            in_channels=in_channels,
            out_channels=out_channels
        )

        self.st_gnn = SpatialTemporalGNN(
            gnn_in_channels=gnn_in_channels,
            gat_out_channels=gat_out_channels,
            gru_hidden_size=gru_hidden_size,
            num_classes=num_classes,
            k_timestamps=k_timestamps
        )

        # Metrics
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, graph):
        node_feature = self.encoder(x)
        batch, n_node, seq_len, feature = node_feature.shape
        graph.x = node_feature.reshape(batch * seq_len * n_node, -1)
        self.st_gnn.seq_length = seq_len
        scores, indices = self.st_gnn(graph)
        return scores, indices

    def training_step(self, batch, batch_idx):
        batch_eeg, batch_graphs, batch_labels = batch
        outputs, _ = self(batch_eeg, batch_graphs)
        loss = self.criterion(outputs, batch_labels)

        # Calculate accuracy
        preds = outputs.argmax(dim=1)
        acc = self.train_accuracy(preds, batch_labels)

        # Log metrics
        self.log('train_loss/batch', loss, prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log('train_acc/batch', acc, prog_bar=True, sync_dist=True, batch_size=self.batch_size)

        return loss

    def on_train_epoch_end(self):
        # Log epoch-level training accuracy
        train_acc = self.train_accuracy.compute()
        self.log('train_acc/epoch', train_acc, prog_bar=True, sync_dist=True)
        self.train_accuracy.reset()

    def validation_step(self, batch, batch_idx):
        batch_eeg, batch_graphs, batch_labels = batch
        outputs, indices = self(batch_eeg, batch_graphs) #top-k 없앰 (outputs, indices)
        loss = self.criterion(outputs, batch_labels)

        # Calculate accuracy
        preds = outputs.argmax(dim=1)
        acc = self.val_accuracy(preds, batch_labels)

        # Log metrics
        self.log('val_loss', loss, prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log('val_acc', acc, prog_bar=True, sync_dist=True, batch_size=self.batch_size)

        return {'val_loss': loss, 'val_acc': acc}

    def on_validation_epoch_end(self):
        val_acc = self.val_accuracy.compute()
        self.log('val_acc/epoch', val_acc, prog_bar=True, sync_dist=True)
        self.val_accuracy.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.learning_rate, weight_decay=0.00001)
        return optimizer