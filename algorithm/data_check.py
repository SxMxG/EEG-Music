import mne
import numpy as np
import time
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

def find_edf(base_dir="."):
    base_path = Path(base_dir)
    data_path = base_path / "data"
    if not data_path.exists():
        data_path = base_path / "algorithm" / "data"
    if not data_path.exists():
        messagebox.showerror("Folder Not Found", f"Could not find 'data' or 'algorithm/data' folder in {base_path}")
        return None
    edf_files = list(data_path.glob("*.edf"))
    if not edf_files:
        messagebox.showinfo("No Files", f"No .edf files found in:\n{data_path}")
        return None
    return edf_files
def parse_filename(edf_path):
    name = edf_path.stem
    parts = [p for p in name.split("_") if p]  # filter empty strings
    day = parts[0]
    month = parts[1]
    year = parts[2]
    if (day == '04' and month == '2'):
        month = '4'
        day = '02'
    try:
        playlist_idx = parts.index("playlist")
        person = parts[playlist_idx + 1]
        # Check if the part after the person name is a number
        after_person = parts[playlist_idx + 2] if playlist_idx + 2 < len(parts) else ""
        number = after_person if after_person.isdigit() else ""
    except (ValueError, IndexError):
        person = "Unknown"
        number = ""
    
    suffix = f" #{number}" if number else ""
    return f"{person}{suffix} {month}/{day}/{year}"
def compare(events,data,log):
    segments = []
    segment_names = []
    segment_durations = []
    discarded = []

    # Pair triggers as (1st,2nd), (3rd,4th), (5th,6th), ...
    def write(*args, **kwargs):
        print(*args, **kwargs)
        print(*args, **kwargs, file=log)
    for i in range(0, len(events) - 1):  # note: -1, and step of 1 not 2
        start_sample = events[i, 0]
        end_sample = events[i + 1, 0]
        name = f"segment_{i+1}_from_trigger_{events[i, 2]}_to_{events[i+1, 2]}"
        
        duration = (float(end_sample) - float(start_sample)) / 300
        if duration > 29.95 and duration < 30.05:
            segment = data[:, start_sample:end_sample]
            segments.append(segment)
            segment_names.append(name)
            segment_durations.append(f"{name}: {duration:.2f}s")
        else:
            discarded.append(f"{name}: {duration:.2f}s")
    write("\n---SEGMENT INFO---")
    write(f"Created a total of {len(segments)+len(discarded)} segments with {len(events)} triggers. "
        f"{len(segments)} used, {len(discarded)} discarded.")
    write("Used segments:", segment_durations)
    write("Discarded segments:", discarded)
    
    

def main():
    edf_files = find_edf()
    total_events = 0
    total_seg = 0
    songs20 = []
    with open("edf_report.txt", "w") as log:
        def p(*args, **kwargs):
            print(*args, **kwargs)
            print(*args, **kwargs, file=log)

        for edf in edf_files:
            p(f"\n{parse_filename(edf)}")
            raw = mne.io.read_raw_edf(edf, preload=True)
            sfreq = int(raw.info["sfreq"])
            events = mne.find_events(raw, stim_channel='Trigger', min_duration=0.0)
            data = raw.get_data()
            if len(events) == 0:
                p("Zero triggers found")
                continue

            
            p(f"Found {len(events)} events, sfreq={sfreq}")
            total_events += len(events)

            # --- Group events into rounds by detecting ID resets ---
            rounds = []
            current_round = []
            for event in events:
                trigger_id = event[2]
                if current_round and trigger_id <= current_round[-1][2]:
                    rounds.append(current_round)
                    current_round = [event]
                else:
                    current_round.append(event)
            if current_round:
                rounds.append(current_round)

            p(f"Detected {len(rounds)} rounds with sizes: {[len(r) for r in rounds]}")

            if len(events) % 2 != 0:
                p(f"Note: odd number of triggers ({len(events)}) - last trigger in a round will be unpaired")

            # --- Slice segments within each round ---
            segments = []
            segment_names = []
            segment_durations = []
            discarded = []
            for round_idx, round_events in enumerate(rounds):
                for i in range(0, len(round_events), 2):
                    start_sample = round_events[i][0]
                    trigger_start = round_events[i][2]

                    if i + 1 < len(round_events):
                        end_sample = round_events[i + 1][0]
                        trigger_end = round_events[i + 1][2]
                        name = f"round_{round_idx+1}_segment_{i//2+1}_trigger_{trigger_start}_to_{trigger_end}"
                    else:
                        # Unpaired final trigger in this round - skip it
                        end_sample = data.shape[1]
                        name = f"round_{round_idx+1}_segment_{i//2+1}_trigger_{trigger_start}_to_session_end"

                    duration = (float(end_sample) - float(start_sample)) / sfreq
                    if 29 < duration < 31:  # accepts both ~30s and ~32s recordings
                        segments.append(data[:, start_sample:end_sample])
                        segment_names.append(name)
                        segment_durations.append(f"{name}: {duration:.2f}s")
                    else:
                        discarded.append(f"{name}: {duration:.2f}s")

            p("\n---SEGMENT INFO---")
            p(f"Created a total of {len(segments)+len(discarded)} segments with {len(events)} triggers. "
              f"{len(segments)} used, {len(discarded)} discarded.")
            p("Used segments:", segment_durations)
            p("Discarded segments:", discarded)
            total_seg += len(segments)
            if(len(segments) + len(discarded) == 20):
                songs20.append(edf)
            p("comparing to just one after the other-----")
            compare(events,data,log)

        p(f'Perfect data files: {songs20}')
        p(f"\nTotal events across all files: {total_events}")
        p(f"Total usable segments: {total_seg}")

if __name__ == "__main__":
    main()