import mne
import numpy as np
import time
from pylsl import StreamInfo, StreamOutlet

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

import tkinter as tk
from tkinter import messagebox
from pathlib import Path

def pick_edf_file(base_dir="."):
    # 1. Check for 'data' folder, then 'algorithm/data'
    base_path = Path(base_dir)
    data_path = base_path / "data"
    if not data_path.exists():
        data_path = Path(base_dir) / "algorithm" / "data"

    if not data_path.exists():
        messagebox.showerror("Folder Not Found", f"Could not find 'data' or 'algorithm/data' folder. in {base_path}")
        return None

    # 2. Find all .edf files
    edf_files = list(data_path.glob("*.edf"))
    if not edf_files:
        messagebox.showinfo("No Files", f"No .edf files found in:\n{data_path}")
        return None

    selected = []

    # 3. Build the Tkinter window
    root = tk.Tk()
    root.title("Select an EDF File")
    root.geometry("500x400")

    tk.Label(root, text=f"Folder: {data_path}", wraplength=480, fg="gray").pack(pady=(10, 0))
    tk.Label(root, text="Select an EDF file:", font=("Arial", 12)).pack(pady=(5, 0))

    listbox = tk.Listbox(root, width=60, height=15, selectmode=tk.SINGLE)
    listbox.pack(padx=10, pady=10)

    for f in edf_files:
        listbox.insert(tk.END, f.name)

    def on_select():
        idx = listbox.curselection()
        if idx:
            selected.append(edf_files[idx[0]])
            root.destroy()
        else:
            messagebox.showwarning("No Selection", "Please click a file first.")

    tk.Button(root, text="Open File", command=on_select, width=20).pack(pady=5)
    root.mainloop()

    return selected[0] if selected else None

def main():
    datapath = pick_edf_file()
    raw = mne.io.read_raw_edf(datapath, preload=True)
    data = raw.get_data()
    sfreq = int(raw.info["sfreq"])
    n_channels = data.shape[0]
    print(f"orginal channel names:{raw.info.ch_names}") #this confirms that channel 8 is a trigger
    print("sampling rate:", sfreq)

    # the amount of events and the number associated  as well as name
    # event_id = {
    # 'event_type_1': 1,
    # 'event_type_2': 2,
    # 'event_type_3': 3,
    # 'event_type_4': 4,
    # }

    # ### Reference to Left Ear channel, default is Pz ###
    # data.set_eeg_reference(ref_channels=['EEG LE-Pz'])
    #
    # ### Standardize channel names ###
    # rename_dict = {}
    # for ch_name in data.ch_names:
    #     if '-Pz' in ch_name:
    #         rename_dict[ch_name] = ch_name.replace('-Pz', '')
    #     if ch_name == "Pz":
    #         rename_dict[ch_name] = "EEG Pz"
    # data.rename_channels(rename_dict)
    # print(f"Updated channel names: {data.info.ch_names}")
    #
    # ### Preprocess with bandpass ###
    # data.filter(l_freq=0.5, h_freq=50.0, picks="eeg")

    ### Find events labeled in the Trigger column ###
    events = mne.find_events(raw, stim_channel='Trigger', min_duration=0.0)

    ###################
    # --- Slice data into segments between trigger pairs ---
    segments = []
    segment_names = []
    segment_durations = []
    discarded = []

    # Pair triggers as (1st,2nd), (3rd,4th), (5th,6th), ...
    for i in range(0, len(events), 2):
        start_sample = events[i, 0]

        if i + 1 < len(events):
            end_sample = events[i + 1, 0]
            name = f"segment_{i//2 + 1}_from_trigger_{events[i, 2]}_to_{events[i+1, 2]}"

        duration = (float(end_sample) - float(start_sample))/300
        if duration > 29.95 and duration < 30.05:
            segment = data[:, start_sample:end_sample]
            segments.append(segment)
            segment_names.append(name)
            segment_durations.append(f"segment_{i//2 + 1}: {duration:.2f}s")
        else:
            discarded.append(f"segment_{i//2 + 1}: {duration:.2f}s")

    print()
    print("---SEGMENT INFO---")
    print(f"Created a total of {len(segments)+len(discarded)} segments with {len(events)} triggers. {len(segments)} segments are used and {len(discarded)} are discarded.")
    print("Used segments:", segment_durations)
    print("Discarded segments:", discarded)
    print()

    ######################

    # print(f'Found {len(events)} events')
    # print(f'Event IDs: {events}')

    # print("n_channels: ", n_channels)
    # print(f"data shape: {len(data.shape)}")
    info = StreamInfo("FakeEEG", "EEG", n_channels, sfreq, "float32", "fake_eeg_stream")
    outlet = StreamOutlet(info)

    chunk_size = 32
    speed = 1
    delay = chunk_size / sfreq * speed

    # ---- GUI setup ----
    app = QtWidgets.QApplication([])
    win = pg.GraphicsLayoutWidget(show=True)
    win.setWindowTitle("EEG Stream Viewer")

    label = win.addLabel(text="Segment: None", row=0, col=0)
    win.nextRow()
    # plot = win.addPlot(title="Channel 1")
    # curve = plot.plot()
    plots = []
    curves = []

    for ch in range(n_channels):
        p = win.addPlot(title=f"Channel {ch + 1}")
        c = p.plot()
        plots.append(p)
        curves.append(c)
        win.nextRow()  # stack vertically

    # buffer = np.zeros(2000)
    buffer = np.zeros((n_channels, 2000))

    # idx = 0
    #
    # while idx < data.shape[1]:
    #
    #     chunk = data[:, idx:idx+chunk_size].T
    #
    #     # send to LSL
    #     for sample in chunk:
    #         outlet.push_sample(sample.tolist())
    #
    #     # update buffer
    #     # buffer = np.roll(buffer, -chunk_size)
    #     # buffer[-chunk_size:] = chunk[:,0]
    #     buffer = np.roll(buffer, -chunk_size, axis=1)
    #     buffer[:, -chunk_size:] = chunk.T
    #
    #     # curve.setData(buffer)
    #     for ch in range(n_channels):
    #         curves[ch].setData(buffer[ch])
    #
    #     QtWidgets.QApplication.processEvents()
    #
    #     time.sleep(delay)
    #
    #     idx += chunk_size

    for seg_idx, segment in enumerate(segments):
        label.setText(segment_names[seg_idx])
        idx = 0

        while idx < segment.shape[1]:
            chunk = segment[:, idx:idx + chunk_size]

            # send to LSL
            for sample in chunk.T:
                outlet.push_sample(sample.tolist())

            # use the actual number of samples in this chunk
            actual_size = chunk.shape[1]

            buffer = np.roll(buffer, -actual_size, axis=1)
            buffer[:, -actual_size:] = chunk

            for ch in range(n_channels):
                curves[ch].setData(buffer[ch])

            QtWidgets.QApplication.processEvents()
            time.sleep(delay)

            idx += chunk_size

    app.exec()

if __name__ == "__main__":
    main()