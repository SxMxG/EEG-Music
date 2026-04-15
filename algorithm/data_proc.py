import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS
from sklearn.cluster import HDBSCAN
from pathlib import Path
from tkinter import messagebox
from hmmlearn import hmm

def verify_alignment(segments, df, p, sfreq=300):
    p(f"\n--- Alignment Check ({len(segments)} segments, {len(df)} labels) ---")
    
    if len(segments) != len(df):
        p(f"WARNING: mismatch! {len(segments)} segments but {len(df)} songs in CSV")
    
    # quick sanity check on raw data
    p(f"Segment shape: {segments[0].shape}")
    p(f"Segment min/max: {segments[0].min():.6f} / {segments[0].max():.6f}")
    p(f"Any non-zero: {np.any(segments[0] != 0)}")
    
    for i, (seg, (_, row)) in enumerate(zip(segments, df.iterrows())):
        features = extract_band_power(seg, sfreq)
        alpha_avg = features[2::5].mean()
        is_flat = np.abs(seg).max() < 1e-10
        flag = " ← FLAT/EMPTY" if is_flat else ""
        p(f"  [{i:2d}] Song #{int(row['Song #']):2d} | {row['Song name']:<30} | "
          f"Likes: {str(row['Likes']):<10} | alpha_avg: {alpha_avg:.6f}{flag}")

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

def extract_band_power_avg(segment, sfreq=300):
    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta":  (13, 30),
        "gamma": (30, 50)
    }
    # channel groupings based on your 7 channels
    # F4,F3 = frontal (emotion), C4,C3 = motor, P4,P3,Pz = parietal (attention)
    regions = {
        "frontal": [0, 5],   # F4, F3
        "central": [1, 4],   # C4, C3
        "parietal": [2, 3, 6] # P4, P3, Pz
    }
    features = []
    for band, (low, high) in bands.items():
        for region_name, ch_idx in regions.items():
            band_powers = []
            for ch in ch_idx:
                freqs, psd = welch(segment[ch], fs=sfreq, nperseg=sfreq*2)
                idx = np.where((freqs >= low) & (freqs <= high))
                band_powers.append(np.mean(psd[idx]))
            features.append(np.mean(band_powers))
    return np.array(features)  # shape: (15,) — 5 bands * 3 regions
def extract_band_power(segment,sfreq=300):
    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta":  (13, 30),
        "gamma": (30, 50)
    }
    # channel groupings based on your 7 channels
    features = []
    for ch in range(segment.shape[0]):  # per channel
        freqs, psd = welch(segment[ch], fs=sfreq, nperseg=sfreq*2)
        for band, (low, high) in bands.items():
            idx = np.where((freqs >= low) & (freqs <= high))
            features.append(np.mean(psd[idx]))  # mean power in band
    return np.array(features)  # shape: (n_channels * 5,)

def build_feature_matrix(segments, sfreq=300, avg=False):
    X = []
    for seg in segments:
        if avg:
            X.append(extract_band_power_avg(seg, sfreq))
        else:
            X.append(extract_band_power(seg, sfreq))
    return np.array(X)
## HMM HELPERS
def fit_hmm(X_scaled, n_states=3):
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=100,
        random_state=42
    )
    model.fit(X_scaled)
    states = model.predict(X_scaled)
    return model, states

def plot_hmm(model, states, df, ax3, ax4,p):
    df = df.copy()
    df["hmm_state"] = states

    # ax3 — state sequence over playlist colored by likes
    likes = df["Likes_num"].values
    ax3.plot(states, linestyle='-', color='gray', alpha=0.4, zorder=1)
    scatter = ax3.scatter(range(len(states)), states, c=likes, cmap="RdYlGn",
                          vmin=-1, vmax=2, zorder=2, s=60)
    plt.colorbar(scatter, ax=ax3, label="Likes")
    ax3.set_yticks(range(model.n_components))
    ax3.set_yticklabels([f"State {i}" for i in range(model.n_components)])
    ax3.set_xlabel("Song #")
    ax3.set_title("HMM Brain States\n(color = likes)")

    # ax4 — mean likes/knows per state
    summary = df.groupby("hmm_state")[["Likes_num", "Knows_num"]].mean()
    x = np.arange(len(summary))
    width = 0.35
    ax4.bar(x - width/2, summary["Likes_num"], width, label="Likes", color="green", alpha=0.7)
    ax4.bar(x + width/2, summary["Knows_num"], width, label="Knows", color="blue", alpha=0.7)
    ax4.set_xticks(x)
    ax4.set_xticklabels([f"State {i}" for i in summary.index])
    ax4.set_title("Mean Likes/Knows\nper HMM State")
    ax4.legend()

    p("\n--- HMM State vs Likes/Knows ---")
    p(df.groupby("hmm_state")[["Likes_num", "Knows_num"]].mean())
    p("\n--- Transition Matrix ---")
    p(pd.DataFrame(
        model.transmat_,
        columns=[f"→{i}" for i in range(model.n_components)],
        index=[f"from {i}" for i in range(model.n_components)]
    ).round(3))
    p("\n--- Songs per State ---")
    for state, group in df.groupby("hmm_state"):
        p(f"\n=== HMM State {state} ===")
        p(group[["Song #", "Song name", "Likes", "Knows"]])

def hdb_fit(X_scaled):
    hdb = HDBSCAN(min_samples=3,min_cluster_size=3)
    hdb.fit(X_scaled)
    return hdb.labels_

def plot_hdb(ax5,labels,X):
    # HDBSCAN scatter
    unique_labels = np.unique(labels)

    for label in unique_labels:
        if label == -1:
            # noise
            ax5.plot(
                X[labels == -1, 0],
                X[labels == -1, 1],
                "k+",
                alpha=0.3,
                label="noise"
            )
        else:
            ax5.plot(
                X[labels == label, 0],
                X[labels == label, 1],
                ".",
                alpha=0.6,
                label=f"cluster {label}"
            )

    ax5.set_title("HDBSCAN Clustering\n(PCA space)")
    ax5.legend()

def main():
    name = input("Name you want to process:")
    with open(f"./algorithm/reports/{name}_report.txt","w",encoding="utf-8") as report:
        def p(*args, **kwargs):
                print(*args, **kwargs)
                print(*args, **kwargs, file=report)
        seg_file, table_file = find_person_file(name)
        segments = np.load(f"{seg_file}")
        p(f"loaded {seg_file} with {len(segments)} segments")
        all_labels = pd.read_csv(f"{table_file}")
        p(f"loaded {table_file} with {len(all_labels)} labels")
        verify_alignment(segments, all_labels,p)

        # feature matrices
        X = build_feature_matrix(segments)              # 35 features — OPTICS
        X_avg = build_feature_matrix(segments, avg=True) # 15 features — HMM

        # scaling
        X_scaled = StandardScaler().fit_transform(X)
        X_avg_scaled = StandardScaler().fit_transform(X_avg)

        # PCA for OPTICS only
        pca = PCA(n_components=0.95)
        X_pca = pca.fit_transform(X_scaled)
        p(f"PCA reduced {X_scaled.shape[1]} -> {X_pca.shape[1]} components")

        # OPTICS on PCA
        clust = OPTICS(min_samples=3, metric='minkowski',xi=0.04)
        clust.fit(X_pca)
        cluster_labels = clust.labels_          # original order — for df
        ordered_labels = clust.labels_[clust.ordering_]  # for reachability plot

        # HMM on region-averaged
        MIN_SONGS_HMM = 10
        run_hmm = len(segments) >= MIN_SONGS_HMM
        if run_hmm:
            n_states = 4 if len(segments) >= 30 else 3
            model, states = fit_hmm(X_avg_scaled, n_states=n_states)
        else:
            p(f"Too few segments ({len(segments)}) for HMM, skipping")

        # build df
        df = all_labels.copy()
        df = df.iloc[:len(cluster_labels)].copy()
        label_map = {
            "None": 0, "dont like": -1, "Dont like": -1,
            "okay": 1, "Okay": 1, "likes": 2, "Likes": 2,
            "Nan": 0, "like": 2, "Like": 2, "yes": 2, "no": -1, "ok": 1
        }
        knows_map = {
            "Know": 2, "Knows": 2, "dont know": 0,
            "Dont know": 0, "Dont know ": 0,
            "Familar": 1, "familar": 1
        }
        df["Likes_num"] = df["Likes"].map(label_map)
        df["Knows_num"] = df["Knows"].map(knows_map)
        df["cluster"] = cluster_labels

        # plots
        space = np.arange(len(X_pca))
        reachability = clust.reachability_[clust.ordering_]

        plt.figure(figsize=(14, 7))
        G = gridspec.GridSpec(2, 4)
        ax1 = plt.subplot(G[0, :])
        ax2 = plt.subplot(G[1, 0])
        ax3 = plt.subplot(G[1, 1])
        ax4 = plt.subplot(G[1, 2])
        ax5 = plt.subplot(G[1, 3])

        # reachability plot
        colors = ["g.", "r.", "b.", "y.", "c."]
        for klass, color in enumerate(colors):
            Xk = space[ordered_labels == klass]
            Rk = reachability[ordered_labels == klass]
            ax1.plot(Xk, Rk, color, alpha=0.3)
        ax1.plot(space[ordered_labels == -1], reachability[ordered_labels == -1], "k.", alpha=0.3)
        ax1.plot(space, np.full_like(space, 2.0, dtype=float), "k-", alpha=0.5)
        ax1.plot(space, np.full_like(space, 0.5, dtype=float), "k-.", alpha=0.5)
        ax1.set_ylabel("Reachability (epsilon distance)")
        ax1.set_title("Reachability Plot")

        # OPTICS scatter on first 2 PCA components
        for klass, color in enumerate(colors):
            Xk = X_pca[cluster_labels == klass]
            ax2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
        ax2.plot(X_pca[cluster_labels == -1, 0], X_pca[cluster_labels == -1, 1], "k+", alpha=0.3)
        ax2.set_title("OPTICS Clustering\n(PCA space)")

        # HMM plots
        if run_hmm:
            plot_hmm(model, states, df, ax3, ax4,p)

        # p cluster info
        p(df)
        df_valid = df[df["cluster"] != -1].copy()
        p(df_valid.groupby("cluster")[["Likes_num", "Knows_num"]].mean())
        for cluster_id, group in df.groupby("cluster"):
            label = "noise" if cluster_id == -1 else cluster_id
            p(f"\n=== Cluster {label} ===")
            p(group.sort_values("Song #")[["Song #", "Song name", "Likes", "Knows"]])
        hdb_labels = hdb_fit(X_pca)
        plot_hdb(ax5,hdb_labels,X_pca)
        plt.tight_layout()
        plt.savefig(f"./algorithm/plots/{name}_plot.png")
        plt.show()
if "__main__" == __name__:
    main()