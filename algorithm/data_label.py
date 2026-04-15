import pandas as pd
from pathlib import Path
from tkinter import messagebox
import numpy as np
def load_song_data(name, base_dir="."):
    base_path = Path(base_dir)
    data_path = base_path / "tables_old"
    if not data_path.exists():
        data_path = base_path / "algorithm" / "tables_old"
    if not data_path.exists():
        messagebox.showerror("Folder Not Found", f"Could not find 'tables_old' or 'algorithm/tables' folder in {base_path}")
        return None
    csv_files = list(data_path.glob(f"*{name}*.csv"))
    if not csv_files:
        messagebox.showinfo("No Files", f"No .csv files found in:\n{data_path}")
        return None
    print(f"found {len(csv_files)} of csv files in the folder")
    song_map = {}
    likes_map = {}
    knows_map = {}
    for csv in csv_files:
        df = pd.read_csv(str(csv))
        print(f"file:{csv}")
        song_map.update(dict(zip(df['Song #'], df['Song name'])))

        if "Likes" in df.columns:
            likes_map.update(dict(zip(df['Song #'], df['Likes'])))
        knows_map.update(dict(zip(df['Song #'], df['Knows'])))
    return song_map, likes_map, knows_map

def main():
    name = input("input a name with Capital first letter:")
    song_map , likes_map,knows_map = load_song_data(name=name)
    df = pd.DataFrame({
        "Song #": list(song_map.keys()),
        "Song name": list(song_map.values()),
        "Likes": [likes_map.get(k,"None") for k in song_map.keys() ],
        "Knows": [knows_map.get(k,"None") for k in song_map.keys()],
    })
    print(df)
    df = df.sort_values("Song #")
    df.to_csv(f"./algorithm/tables/{name}_song_list.csv",index=False)

if __name__ == "__main__":
    main()