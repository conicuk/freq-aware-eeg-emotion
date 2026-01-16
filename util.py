import torch
import numpy as np
import scipy.io as sio
from scipy import signal
import os
from tqdm import tqdm
import torch.nn.functional as F


def load_data(data_dir):
    X_all = []
    y_all = []

    for subject in tqdm(range(1, 17), desc="Loading subjects"):
        file_name = f"HFsim_{subject:02d}_Cut.mat"
        file_path = os.path.join(data_dir, file_name)

        mat = sio.loadmat(file_path)

        happy_trials = mat['HFsim_EEGcut_happy']
        neutral_trials = mat['HFsim_EEGcut_neutral']
        sad_trials = mat['HFsim_EEGcut_sad']

        X_subject = np.concatenate([happy_trials, neutral_trials, sad_trials], axis=2)

        n_happy = happy_trials.shape[2]
        n_neutral = neutral_trials.shape[2]
        n_sad = sad_trials.shape[2]

        y_subject = np.concatenate([
            np.zeros(n_happy),
            np.ones(n_neutral),
            2 * np.ones(n_sad)
        ])

        X_all.append(X_subject)
        y_all.append(y_subject)

    return np.concatenate(X_all, axis=2), np.concatenate(y_all)


def extract_frequency_bands(data, fs=250):
    freq_bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 120)
    }

    window_size = 5  # 0.3초 (250Hz * 0.3s = 75 samples -> 75일때 얘기)
    n_channels, n_timepoints = data.shape
    n_windows = n_timepoints // window_size
    features = np.zeros((n_channels, n_windows, len(freq_bands)))

    for band_idx, (band_name, (low_freq, high_freq)) in enumerate(freq_bands.items()):
        # Design bandpass filter
        b, a = signal.butter(4, [low_freq, high_freq], btype='bandpass', fs=fs)

        # Apply filter
        filtered = signal.filtfilt(b, a, data, axis=1)

        # Non-overlapping windows
        for t in range(n_windows):
            start_idx = t * window_size
            end_idx = (t + 1) * window_size
            window = filtered[:, start_idx:end_idx]
            variance = np.var(window, axis=1)
            de = 0.5 * np.log2(2 * np.pi * np.e * variance + 1e-10)
            features[:, t, band_idx] = de

    return features


def prepare_dataset(data_dir):
    print("Loading data...")
    X, y = load_data(data_dir)

    # Reshape to (n_trials, n_channels, time_points)
    n_trials = X.shape[2]
    X = np.transpose(X, (2, 0, 1))

    # Extract features for each trial
    features_list = []
    for trial in tqdm(X, desc="Processing trials"):
        trial_features = extract_frequency_bands(trial)
        features_list.append(trial_features)

    # Stack all trials
    features = np.stack(features_list)

    print(f"Final feature shape: {features.shape}")
    return features, y


def average_pool_plv(plv_data):
    # plv_data : (num_samples, n_channel, n_channel, time_points) -> (4751, 63, 63, 1875)
    with torch.no_grad():
        batch_size, n_nodes_i, n_nodes_j, time_points = plv_data.shape

        plv_data = plv_data.reshape(batch_size * n_nodes_i, 1, n_nodes_j, time_points)

        plv_data = F.avg_pool2d(
            plv_data,
            kernel_size=(1,3),
            stride=(1,2),
            padding=0
        )

        print("plv shape", plv_data.shape)  # -> (4751 * 63, 1, 63, 937)

    return plv_data.reshape(batch_size, n_nodes_i, n_nodes_j, -1)


if __name__ == "__main__":
    data_dir = "/home/coni/CONIRepo/Seoyeon/EmoNet_attention/HFsim"
    X, y = prepare_dataset(data_dir)

    print("Feature shape:", X.shape)  # Should be (4751, 63, n_windows, 5)
    print("Label shape:", y.shape)

    # Save the preprocessed data
    np.save('HFsim_pre_features_v1.npy', X)
    np.save('HFsim_pre_labels_v1.npy', y)

