import mne
import numpy as np
import time
from pylsl import StreamInfo, StreamOutlet

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

def main():

    raw = mne.io.read_raw_edf("./data/04_2_2026_playlist_Ella_raw.edf", preload=True)
    raw.filter(l_freq=1.0, h_freq=50.0)
    data = raw.get_data()
    sfreq = int(raw.info["sfreq"])
    n_channels = data.shape[0]

    info = StreamInfo("FakeEEG", "EEG", n_channels, sfreq, "float32", "fake_eeg_stream")
    outlet = StreamOutlet(info)

    chunk_size = 32
    speed = 1
    delay = chunk_size / sfreq * speed

    # ---- GUI setup ----
    app = QtWidgets.QApplication([])
    win = pg.GraphicsLayoutWidget(show=True)
    win.setWindowTitle("EEG Stream Viewer")

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

    idx = 0

    while idx < data.shape[1]:

        chunk = data[:, idx:idx+chunk_size].T

        # send to LSL
        for sample in chunk:
            outlet.push_sample(sample.tolist())

        # update buffer
        # buffer = np.roll(buffer, -chunk_size)
        # buffer[-chunk_size:] = chunk[:,0]
        buffer = np.roll(buffer, -chunk_size, axis=1)
        buffer[:, -chunk_size:] = chunk.T

        # curve.setData(buffer)
        for ch in range(n_channels):
            curves[ch].setData(buffer[ch])

        QtWidgets.QApplication.processEvents()

        time.sleep(delay)

        idx += chunk_size

    app.exec()

if __name__ == "__main__":
    main()