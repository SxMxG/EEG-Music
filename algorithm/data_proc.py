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
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import json
import time
sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id="clientid",
        client_secret="secret"
    ),
)
def analyze_states_with_spotify(df, df_spotify, states, p):
    df = df.copy()
    df["hmm_state"] = states
    
    # merge spotify data in
    df_merged = df.merge(df_spotify, left_on="Song name", right_on="song", how="left")
    
    # explode genres so each genre is its own row for counting
    df_exploded = df_merged.explode("genres")
    
    p("\n--- Spotify Analysis per HMM State ---")
    for state, group in df_merged.groupby("hmm_state"):
        p(f"\n=== State {state} ===")
        
        # top genres
        state_genres = df_exploded[df_exploded["hmm_state"] == state]["genres"]
        top_genres = state_genres.value_counts().head(5)
        p(f"  Top genres: {dict(top_genres)}")
        
        # avg popularity
        avg_pop = group["popularity"].mean()
        p(f"  Avg popularity: {avg_pop:.1f}/100")
        
        # avg likes
        avg_likes = group["Likes_num"].mean()
        p(f"  Avg likes: {avg_likes:.2f} (-1=dislike, 1=okay, 2=like)")
        
        # artists
        artists = group["artist"].value_counts().head(3)
        p(f"  Top artists: {dict(artists)}")

def load_or_fetch_features(song_list, cache_path="song_features_cache.json"):
    cache_path = Path(cache_path)
    
    # load existing cache
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
    else:
        cache = {}
    
    # only fetch songs not already in cache
    missing = [s for s in song_list if s not in cache]
    if missing:
        print(f"Fetching {len(missing)} new songs from Spotify...")
        for song in missing:
            feat = get_song_features(song)
            cache[song] = feat if feat else {"song": song, "artist": None, "genres": [], "popularity": None}
            time.sleep(0.5)  # be nice to the API
        
        # save updated cache
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        print(f"Saved cache to {cache_path}")
    else:
        print("All songs found in cache, no API calls needed")
    
    return [cache[s] for s in song_list]
def get_song_features(song_name):
    results = sp.search(q=song_name, type="track", limit=1)
    if not results["tracks"]["items"]:
        return None
    track = results["tracks"]["items"][0]
    artist_id = track["artists"][0]["id"]
    artist = sp.artist(artist_id)
    
    return {
        "song": song_name,
        "artist": track["artists"][0]["name"],
        "genres": artist["genres"],
        "popularity": track["popularity"],
    }
def verify_alignment(segments, df, p, sfreq=300):
    # add this to verify_alignment temporarily
    seg = segments[0]
    freqs, psd = welch(seg[0], fs=sfreq, nperseg=sfreq*2)
    p(f"freqs range: {freqs[0]:.2f} - {freqs[-1]:.2f}")
    print(f"psd range: {psd.min():.8f} - {psd.max():.8f}")

    # check alpha band specifically
    alpha_idx = np.where((freqs >= 8) & (freqs <= 13))
    p(f"alpha idx: {alpha_idx}")
    print(f"alpha power: {np.mean(psd[alpha_idx]):.8f}")

    features = extract_band_power(seg, sfreq)
    p(f"features shape: {features.shape}")
    p(f"features range: {features.min():.8f} - {features.max():.8f}")
    p(f"first 10 features: {features[:10]}")

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
def fit_hmm_best(X_scaled, p,n_states=3, n_restarts=10):
    best_model = None
    best_balance = float('inf')
    
    for seed in range(n_restarts):
        model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag",
                                 n_iter=100, random_state=seed)
        model.fit(X_scaled)
        states = model.predict(X_scaled)
        
        counts = np.bincount(states, minlength=n_states)
        # measure imbalance — lower is more even
        imbalance = counts.max() - counts.min()
        
        if imbalance < best_balance:
            best_balance = imbalance
            best_model = model
            best_states = states
    unique, counts = np.unique(best_states, return_counts=True)
    p("\n--- HMM State Distribution ---")
    for state, count in zip(unique, counts):
        p(f"  State {state}: {count} songs ({count/len(best_states)*100:.1f}%)")
    return best_model, best_states

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

def interpret_hmm_states(model, states, segments, p, sfreq=300):
    bands = ["delta", "theta", "alpha", "beta", "gamma"]
    regions = ["frontal", "central", "parietal"]
    
    # build column names matching feature order
    cols = [f"{band}_{region}" for band in bands for region in regions]
    
    # build feature matrix
    X_avg = np.array([extract_band_power_avg(seg, sfreq) for seg in segments])
    
    # normalize so we can compare relative activation
    X_norm = StandardScaler().fit_transform(X_avg)
    
    df_features = pd.DataFrame(X_norm, columns=cols)
    df_features["hmm_state"] = states
    
    summary = df_features.groupby("hmm_state").mean()
    
    p("\n--- HMM State Brain Pattern (z-scored, higher = more active) ---")
    p(summary.round(3).to_string())
    
    # print interpretation per state
    p("\n--- State Interpretations ---")
    for state in sorted(df_features["hmm_state"].unique()):
        row = summary.loc[state]
        top = row.nlargest(3)
        bottom = row.nsmallest(3)
        p(f"\nState {state}:")
        p(f"  Most active:   {', '.join([f'{k} ({v:.2f})' for k,v in top.items()])}")
        p(f"  Least active:  {', '.join([f'{k} ({v:.2f})' for k,v in bottom.items()])}")

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

    ax5.set_title("HDBSCAN Clustering")
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
        mask_dup = ~all_labels.duplicated(subset="Song name", keep="last")
        mask_skip = ~all_labels["Song name"].str.contains("dont use this one", case=False, na=False)
        mask_nan = ~all_labels["Likes"].str.contains("None",na=True)
        mask = mask_dup & mask_skip & mask_nan
        all_labels = all_labels[mask].reset_index(drop=True)
        song_features = load_or_fetch_features(
            all_labels["Song name"].tolist(),
            cache_path="./algorithm/song_features_cache.json"
        )
        df_spotify = pd.DataFrame(song_features)
        p(df_spotify[["song", "artist", "genres", "popularity"]])
        segments = segments[mask.values]
        verify_alignment(segments, all_labels,p)
        # feature matrices
        X = build_feature_matrix(segments)              # 35 features — OPTICS
        X_avg = build_feature_matrix(segments, avg=True) # 15 features — HMM

        # scaling
        X_scaled = StandardScaler().fit_transform(X)
        X_avg_scaled = StandardScaler().fit_transform(X_avg)

        # # PCA for OPTICS only
        # pca = PCA(n_components=0.95)
        # X_pca = pca.fit_transform(X_scaled)
        # p(f"PCA reduced {X_scaled.shape[1]} -> {X_pca.shape[1]} components")

        # OPTICS on PCA
        clust = OPTICS(min_samples=3, metric='minkowski',xi=0.04)
        clust.fit(X_scaled)
        cluster_labels = clust.labels_          # original order — for df
        ordered_labels = clust.labels_[clust.ordering_]  # for reachability plot

        # HMM on region-averaged
        MIN_SONGS_HMM = 10
        run_hmm = len(segments) >= MIN_SONGS_HMM
        if run_hmm:
            n_states = 5 if len(segments) > 20 else 3
            model, states = fit_hmm_best(X_avg_scaled,p,n_states=n_states)
        else:
            p(f"Too few segments ({len(segments)}) for HMM, skipping")

        # build df
        df = all_labels.copy()
        df = df.iloc[:len(cluster_labels)].copy()
        label_map = {
            "None": 0, "dont like": -1, "Dont like": -1, "don't like":-1, 
            "okay": 1, "Okay": 1, "likes": 2, "Likes": 2,
            "Nan": 0, "like": 2, "Like": 2, "yes": 2, "no": -1, "ok": 1
        }
        knows_map = {
            "Know": 2, "Knows": 2, "know":2,
            "dont know": -1,"Dont know": -1, "Dont know ": -1,"don't know":-1,
            "Familar": 1, "familar": 1
        }
        df["Likes_num"] = df["Likes"].map(label_map)
        df["Knows_num"] = df["Knows"].map(knows_map)
        df["cluster"] = cluster_labels

        # plots
        space = np.arange(len(X_scaled))
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


        # HMM plots
        if run_hmm:
            plot_hmm(model, states, df, ax3, ax4,p)
            interpret_hmm_states(model,states,segments,p)
            analyze_states_with_spotify(df,df_spotify,states,p)
        # p cluster info
        p(df)
        df_valid = df[df["cluster"] != -1].copy()
        p(df_valid.groupby("cluster")[["Likes_num", "Knows_num"]].mean())
        for cluster_id, group in df.groupby("cluster"):
            label = "noise" if cluster_id == -1 else cluster_id
            p(f"\n=== Cluster {label} ===")
            p(group.sort_values("Song #")[["Song #", "Song name", "Likes", "Knows"]])
        hdb_labels = hdb_fit(X_scaled)

        pca_viz = PCA(n_components=2)
        X_viz = pca_viz.fit_transform(X_scaled)

        # then in scatter
        for klass, color in enumerate(colors):
            Xk = X_viz[cluster_labels == klass]
            ax2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
        ax2.plot(X_viz[cluster_labels == -1, 0], X_viz[cluster_labels == -1, 1], "k+", alpha=0.3)

        # and for hdbscan
        plot_hdb(ax5, hdb_labels, X_viz)  # pass X_viz not X_scaled
        # Hdb cluster  report 
        df["hdb_cluster"] = hdb_labels
        p("\n--- HDBSCAN Results ---")
        df_hdb_valid = df[df["hdb_cluster"] != -1]
        p(df_hdb_valid.groupby("hdb_cluster")[["Likes_num", "Knows_num"]].mean())
        for cluster_id, group in df.groupby("hdb_cluster"):
            label = "noise" if cluster_id == -1 else cluster_id
            p(f"\n=== HDBSCAN Cluster {label} ===")
            p(group.sort_values("Song #")[["Song #", "Song name", "Likes", "Knows"]])

        #finish up
        plt.tight_layout()
        plt.savefig(f"./algorithm/plots/{name}_plot.png")
        plt.show()
if "__main__" == __name__:
    main()