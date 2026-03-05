import mne, time
from pylsl import StreamInfo, StreamOutlet
import numpy as np

raw = mne.io.read_raw_edf("./data", preload=True)
data = raw.get_data()
sfreq = int(raw.info["sfreq"])
n_channels = data.shape[0]

info = StreamInfo("FakeEEG", "EEG", n_channels, sfreq, "float32", "fake_eeg_stream")
outlet = StreamOutlet(info)

chunk_size = 32
speed = 10
delay = chunk_size / sfreq * speed

idx = 0
while idx < data.shape[1]:
    chunk = data[:, idx:idx+chunk_size].T
    for sample in chunk:
        outlet.push_sample(sample.tolist())
    time.sleep(delay)
    print(f"Sent chunk starting at: {idx}")
    idx += chunk_size