import mne
import numpy as np
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
    print(f"found {len(edf_files)} of edf files in the folder")
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
    info = [person,suffix,month,day,year]
    return f"{person}{suffix} {month}/{day}/{year}"

def compare(events, data, log, sfreq=300):
    segments = []
    segment_names = []
    segment_durations = []
    discarded = []

    def write(*args, **kwargs):
        print(*args, **kwargs)
        print(*args, **kwargs, file=log)

    for i in range(0, len(events) - 1):  # all except last
        start_sample = events[i, 0]
        end_sample = events[i + 1, 0]
        name = f"segment_{len(segments)+1}_from_trigger_{events[i, 2]}_to_{events[i+1, 2]}"
        
            
        duration = (float(end_sample) - float(start_sample)) / sfreq
        if(events[i,2] == 15 and events[i+1,2] == 1 and duration > 30):
            songs2 = duration > 60
            write("restart round so lost a trigger")
            end_sample = start_sample + 30 * sfreq
            duration = (float(end_sample) - float(start_sample)) / sfreq
            segment = data[:,start_sample:end_sample]
            segments.append(segment)
            name = f"segment_{i+1}_from_trigger_{events[i, 2]}_to_30s_plus"
            segment_names.append(name)
            segment_durations.append(f"{name}:{duration:.2f}")
            if(songs2):
                write("found duration for 2 songs so splicing")
                start_sample = end_sample
                end_sample = start_sample + 30 * sfreq
                duration = (float(end_sample) - float(start_sample)) / sfreq
                segment = data[:,start_sample:end_sample]
                segments.append(segment)
                name = f"segment_{len(segments)}_from_30s_plus_to_trigger_{events[i+1, 2]}"
                segment_names.append(name)
                segment_durations.append(f"{name}:{duration:.2f}")
        elif 29 < duration < 34:
            if(duration * sfreq != 9000):
                write(f"{name}:{duration*sfreq:.2f} fixed to -> 30s and 9000 samples")
                end_sample = start_sample + 30 * sfreq
                duration = (float(end_sample) - float(start_sample)) / sfreq 
            segment = data[:, start_sample:end_sample]
            segments.append(segment)
            segment_names.append(name)
            segment_durations.append(f"{name}: {duration:.2f}s")
        else:
            discarded.append(f"{name}: {duration:.2f}s")

    # Handle the last trigger — no next trigger, so use +30s
    last_sample = events[-1, 0]
    last_id = events[-1, 2]
    end_sample = last_sample + 30 * sfreq
    end_sample = min(end_sample, data.shape[1])  # don't go past end of data
    name = f"segment_{len(events)}_from_trigger_{last_id}_to_plus_30s"
    duration = (float(end_sample) - float(last_sample)) / sfreq
    if 29 < duration:
        segment = data[:, last_sample:end_sample]
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
    return segments, segment_durations
    
def find_csv(base_dir="."):
    base_path = Path(base_dir)
    data_path = base_path / "tables"
    if not data_path.exists():
        data_path = base_path / "algorithm" / "tables"
    if not data_path.exists():
        messagebox.showerror("Folder Not Found", f"Could not find 'data' or 'algorithm/data' folder in {base_path}")
        return None
    csv_files = list(data_path.glob("*.csv"))
    if not csv_files:
        messagebox.showinfo("No Files", f"No .csv files found in:\n{data_path}")
        return None
    print(f"found {len(csv_files)} of csv files in the folder")
    return csv_files



def main():
    edf_files = find_edf()
    total_events = 0
    total_seg = 0
    songs20 = []
    all_segments = []
    all_labels = []
    per_info = {}
    with open("edf_report.txt", "w") as log:
        def p(*args, **kwargs):
            print(*args, **kwargs)
            print(*args, **kwargs, file=log)

        for edf in edf_files:
            info = parse_filename(edf)
            p(f"\n{info}")
            pname = info.split(" ")[0]
            if(pname not in per_info):
                per_info[pname] = {"songs #" : 0,"segments":[]}
            raw = mne.io.read_raw_edf(edf, preload=True)
            raw.set_eeg_reference(ref_channels=['EEG LE-Pz'])
            channels = ['EEG F4', 'EEG C4', 'EEG P4', 'EEG P3', 'EEG C3', 'EEG F3', 'EEG Pz']
            sfreq = int(raw.info["sfreq"])
            events = mne.find_events(raw, stim_channel='Trigger', min_duration=0.0)
            rename_dict = {}
            for ch_name in raw.ch_names:
                if '-Pz' in ch_name:
                    rename_dict[ch_name] = ch_name.replace('-Pz', '')
                if ch_name == "Pz":
                    rename_dict[ch_name] = "EEG Pz"
            raw.rename_channels(rename_dict)
            raw = raw.pick(channels)
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
                        end_sample = start_sample + 30 * sfreq
                        end_sample = min(end_sample,data.shape[1])
                        name = f"round_{round_idx+1}_segment_{i//2+1}_trigger_{trigger_start}_to_plus_30s"
                        
                        if round_idx + 1 < len(rounds):
                            next_round_start = rounds[round_idx + 1][0][0]
                            gap = (next_round_start - start_sample) / sfreq
                            if next_round_start < end_sample:
                                p(f"  WARNING: trigger_{trigger_start}+30s overlaps next round "
                                f"(next round starts {gap:.2f}s after trigger_{trigger_start}, cutting there)")
                                end_sample = next_round_start
                            else:
                                p(f"  OK: next round starts {gap:.2f}s after trigger_{trigger_start}, 30s window is clean")

                    duration = (float(end_sample) - float(start_sample)) / sfreq
                    if 29 < duration < 31:  # accepts both ~30s and ~31s recordings
                        if(duration * sfreq != 9000):
                            p(f"{name}:{duration:.2f} fixed to -> 30s and 9000 samples")
                            end_sample = start_sample + 30 * sfreq
                            duration = (float(end_sample) - float(start_sample)) / sfreq 
                        seg = data[:, start_sample:end_sample]
                        segments.append(seg)
                        segment_names.append(name)
                        segment_durations.append(f"{name}: {duration:.2f}s")
                        all_segments.append(seg)
                        all_labels.append(f"{parse_filename(edf)}_{name}")
                    else:
                        discarded.append(f"{name}: {duration:.2f}s")

            p("\n---SEGMENT INFO---")
            p(f"Created a total of {len(segments)+len(discarded)} segments with {len(events)} triggers. "
              f"{len(segments)} used, {len(discarded)} discarded.")
            p("Used segments:", segment_durations)
            p("Discarded segments:", discarded)
            if(pname == "Joonha" or (pname == "Yichen" and len(segments) < 20) or pname == "Andrew"):
                if(info.count("3") > 1 and pname =="Yichen"):
                    p("Yichen 3 found")
                p("comparing to just one after the other-----")
                comp_seg,comp_names = compare(events,data,log)
                p(f"\n###using one after the other counting {len(comp_seg)}####")

                segments = comp_seg
                segment_durations = comp_names
            per_info[pname]["songs #"] += len(segments)
            p(f"adding {len(segments)} amount of segments to {pname}")
            per_info[pname]["segments"].extend(segments)
            total_seg += len(segments)
            if(len(segments) >= 20):
                songs20.append(edf)

        p(f'Perfect data files: {[str(f) for f in songs20]}')
        p(f"\nTotal events across all files: {total_events}")
        p(f"Total usable songs: {total_seg}")
        p(f"\n--- PER PERSON SUMMARY ---")
        for person, info in per_info.items():
            p(f"{person}: {info['songs #']} songs , {len(info["segments"])} segments loaded")
            np.save(f"algorithm/segments/{person}_segments.npy",np.array(info["segments"]))
        # np.save("segments.npy",np.array(all_segments))
        # np.save("segment_labels.npy",np.array(all_labels))
        # p(f"Saved{len(all_segments)} segmemnts into segments.npy")

if __name__ == "__main__":
    main()