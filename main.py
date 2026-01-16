from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from dataset import EEGDataset,CombinedDataset,BrainGraphDataset
from torch.utils.data import DataLoader
from model import custom_collate_fn
from train import LightningEnSTGNN
from pytorch_lightning import seed_everything
import pytorch_lightning as pl


seed_everything(42)

data_dir = "/home/coni/CONIRepo/Seoyeon/EmoNet_attention/"
features = torch.tensor(np.load(data_dir + 'HFsim_pre_features_v1.npy'), dtype=torch.float32)
labels = np.load(data_dir + 'HFsim_pre_labels_v1.npy')
plv_data = np.load(data_dir + 'HFsim/HFsim_PLV_all_fre_pooling.npy')

# Model hyperparameters
hparams = {
    'seq_length': 375,
    'feature_size': 5,
    'in_channels': 63,
    'out_channels': 63,
    'gnn_in_channels': 5,
    'gat_out_channels': 32,
    'gru_hidden_size': 64,
    'num_classes': 3,
    #'k_timestamps': 5,
    'learning_rate': 0.0001,
    'batch_size': 55
}

# K-fold Cross Validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
    print(f"\nFold {fold + 1}/10")

    # Create datasets
    train_dataset = CombinedDataset(
        EEGDataset(features[train_idx], labels[train_idx]),
        BrainGraphDataset(plv_data[train_idx], len(train_idx))
    )

    val_dataset = CombinedDataset(
        EEGDataset(features[val_idx], labels[val_idx]),
        BrainGraphDataset(plv_data[val_idx], len(val_idx))
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=hparams['batch_size'],
        collate_fn=custom_collate_fn,
        num_workers=4,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=hparams['batch_size'],
        collate_fn=custom_collate_fn,
        num_workers=4
    )

    # Initialize model
    model = LightningEnSTGNN(seq_length=hparams["seq_length"],
                             feature_size=hparams["feature_size"],
                             in_channels=hparams["in_channels"],
                             out_channels=hparams["out_channels"],
                             gnn_in_channels=hparams["gnn_in_channels"],
                             gat_out_channels=hparams["gat_out_channels"],
                             gru_hidden_size=hparams["gru_hidden_size"],
                             num_classes=hparams["num_classes"],
                             #k_timestamps=hparams["k_timestamps"],
                             learning_rate=hparams["learning_rate"],
                             batch_size=hparams["batch_size"])

    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=30,
        mode='min'
    )

    model_name = ("HFsim" + f"lr{hparams['learning_rate']}_"
                            f"batch{hparams['batch_size']}_gru{hparams['gru_hidden_size']}_"
                            f"gat{hparams['gat_out_channels']}_lr{hparams['learning_rate']}")

    checkpoint_callback = ModelCheckpoint(
        dirpath=f'{data_dir}/HFsim_checkpoint/{model_name}',
        filename=f'fold_{fold}' + '_acc{val_acc:.4f}',
        monitor='val_acc',
        mode='max',
        auto_insert_metric_name=False
    )

    # Logger
    logger = TensorBoardLogger('lightning_logs', name=model_name)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=[early_stopping, checkpoint_callback],
        logger=logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=[0, 1, 2]
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)

    # Store results
    fold_results.append(checkpoint_callback.best_model_score.item())

# Print final results
mean_acc = np.mean(fold_results)
std_acc = np.std(fold_results)
print(f"\nFinal Results: {mean_acc:.4f} ± {std_acc:.4f}")



