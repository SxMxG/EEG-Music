import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS
from pathlib import Path
from tkinter import messagebox

def find_person_file(name,base_dir="."):
    base_path = Path(base_dir)
    seg_path = base_path / "segments"
    table_path = base_path / "tables"
    if not table_path.exists():
        table_path = base_path / "algorithm" / "tables"
    if not seg_path.exists():
        seg_path = base_path / "algorithm" / "segments"
    if not table_path.exists():
        messagebox.showerror("Folder Not Found", f"Could not find 'tables' or 'algorithm/tables' folder in {base_path}")
        return None
    if not seg_path.exists():
        messagebox.showerror("Folder Not Found", f"Could not find 'segments' or 'algorithm/segments' folder in {base_path}")
        return None
    seg_file = list(seg_path.glob(f"*{name}*.npy"))
    table_file = list(table_path.glob(f"*{name}*.csv"))
    if not seg_file:
        messagebox.showinfo("No Segments", f"No files found in:\n{seg_path}")
        return None
    if not table_file:
        messagebox.showinfo("No table",f"No files found in:\n{table_path}")
    print(f"found both files in the folder")
    return seg_file[0] , table_file[0]

def extract_band_power(segment, sfreq=300):
    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta":  (13, 30),
        "gamma": (30, 50)
    }
    features = []
    for ch in range(segment.shape[0]):  # per channel
        freqs, psd = welch(segment[ch], fs=sfreq, nperseg=sfreq*2)
        for band, (low, high) in bands.items():
            idx = np.where((freqs >= low) & (freqs <= high))
            features.append(np.mean(psd[idx]))  # mean power in band
    return np.array(features)  # shape: (n_channels * 5,)

def build_feature_matrix(segments, sfreq=300):
    X = []
    for seg in segments:
        X.append(extract_band_power(seg, sfreq))
    return np.array(X)

def main():
    name = input("Name you want to process:")
    seg_file,table_file = find_person_file(name)
    segments = np.load(f"{seg_file}")
    print(f"loaded {seg_file} with {len(segments)} segments")
    all_labels = pd.read_csv(f"{table_file}")
    print(f"loaded {table_file} with {len(all_labels)} labels")
    X = build_feature_matrix(segments)
    print(segments)
    print(all_labels)
    X_scaled = StandardScaler().fit_transform(X)
    print(X_scaled)
    clust = OPTICS(min_samples=3,metric='minkowski')
    clust.fit(X_scaled)
    labels = clust.labels_  # -1 = noise, 0,1,2... = clusters
    print(labels)

    space = np.arange(len(X_scaled))
    reachability = clust.reachability_[clust.ordering_]
    labels = clust.labels_[clust.ordering_]

    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(2, 3)
    ax1 = plt.subplot(G[0, :])
    ax2 = plt.subplot(G[1, 0])
    ax3 = plt.subplot(G[1, 1])
    ax4 = plt.subplot(G[1, 2])

    # Reachability plot
    colors = ["g.", "r.", "b.", "y.", "c."]
    for klass, color in enumerate(colors):
        Xk = space[labels == klass]
        Rk = reachability[labels == klass]
        ax1.plot(Xk, Rk, color, alpha=0.3)
    ax1.plot(space[labels == -1], reachability[labels == -1], "k.", alpha=0.3)
    ax1.plot(space, np.full_like(space, 2.0, dtype=float), "k-", alpha=0.5)
    ax1.plot(space, np.full_like(space, 0.5, dtype=float), "k-.", alpha=0.5)
    ax1.set_ylabel("Reachability (epsilon distance)")
    ax1.set_title("Reachability Plot")

    # OPTICS
    colors = ["g.", "r.", "b.", "y.", "c."]
    for klass, color in enumerate(colors):
        Xk = X[clust.labels_ == klass]
        ax2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    ax2.plot(X[clust.labels_ == -1, 0], X[clust.labels_ == -1, 1], "k+", alpha=0.1)
    ax2.set_title("Automatic Clustering\nOPTICS")
    df = all_labels.copy()
    df = df.iloc[:len(labels)].copy()
    label_map = {
        "None":0,
        "dont like": -1,
        "Dont like": -1,
        "okay": 1,
        "Okay": 1,
        "likes": 2,
        "Likes": 2,
        "Nan": 0,
        "like":2,
        "Like":2,
        "yes":2,
        "no":-1

    }
    knows_map = {
        "Know":2,
        "dont know":0,
        "Dont know": 0,
        "Dont know ":0,
        "Familar":1,
        "familar":1
    }

    df["Likes_num"] = df["Likes"].map(label_map)
    df["Knows_num"] = df["Knows"].map(knows_map)
    df["cluster"] = labels
    print(df)
    df_valid = df[df["cluster"] != -1].copy()
    print(df_valid.groupby("cluster")[["Likes_num", "Knows_num"]].mean())
    plt.tight_layout()
    plt.show()
if "__main__" == __name__:
    main()