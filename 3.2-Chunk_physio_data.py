import os
import gc
import mne
import numpy as np
import pandas as pd
import argparse
import logging

from mne_bids import BIDSPath
from func.configs import DECIM, DURATION_CHAR, DURATION_WORD, MAX_TIMELAG


# Directories
BIDS_ROOT = "./data/bids/SMN4Lang"
PREPROC_ROOT = os.path.join(BIDS_ROOT, "derivatives/preprocessed_data")
SAVE_ROOT = "./data/saved"


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )

    parser = argparse.ArgumentParser(description="A script to chunk MEG data for chars and words.")
    parser.add_argument("--char", help="Chunk MEG data into char-level.", action="store_true")
    parser.add_argument("--word", help="Chunk MEG data into word-level.", action="store_true")
    args = parser.parse_args()

    save_root = os.path.join(SAVE_ROOT, "ChunkedPhysioData/sensor_level")

    if args.char:
        logging.info("--- Chunking Char-Level MEG Data ---")
        DURATION = DURATION_CHAR
        trans_dir = os.path.join(SAVE_ROOT, "char_transcription")
        for sub in ["sub-01","sub-02","sub-03","sub-04","sub-05",
                    "sub-06","sub-07","sub-10","sub-11","sub-12"]:
            save_dir = os.path.join(save_root, "char_chunkphysio", sub)
            os.makedirs(save_dir, exist_ok=True)
            for run in range(1, 61):
                logging.info(f"Processing Run {run} for Subject '{sub}'... ")
                bids_path = BIDSPath(subject=sub.split("-")[-1],
                                    task="RDR", run=f"{run}", suffix="meg", datatype="MEG",
                                    root=PREPROC_ROOT, extension=".fif", check=False)
                print(bids_path)

                raw = mne.io.read_raw_fif(bids_path, preload=True)
                original_sfreq = raw.info["sfreq"]
                new_sfreq = original_sfreq / DECIM
                raw_resampled = raw.copy().resample(new_sfreq)
                
                events = mne.find_events(raw_resampled, shortest_event=1)
                sfreq = raw_resampled.info["sfreq"]
                window_samples = int(DURATION * sfreq)
                offset = int(MAX_TIMELAG * new_sfreq)
                
                print(f"Original sfreq: {original_sfreq}, New sfreq: {sfreq}")
                print(f"Window samples for {int(DURATION * 1000)} ms: {window_samples}")
                print(f"Offset time lag samples for {int(MAX_TIMELAG * 1000)} ms: {offset}")

                start_time = (events[0, 0] - raw_resampled.first_samp) / sfreq
                end_time = (events[1, 0] - raw_resampled.first_samp) / sfreq
                
                raw_cropped = raw_resampled.copy().crop(tmin=start_time, tmax=end_time + DURATION + MAX_TIMELAG).pick(picks="meg")
                crop_data = raw_cropped.get_data()
                print(f"Cropped data shape: {crop_data.shape}")
                
                duration = end_time - start_time + DURATION + MAX_TIMELAG
                timevec = np.linspace(0, duration, crop_data.shape[-1])

                physio_data = list()
                trans_file = os.path.join(trans_dir, f"story_{run}.csv")
                start_times = pd.read_csv(trans_file)["start"].tolist()
                for start in start_times:
                    start_idx = np.argmin(np.abs(timevec - start))
                    end_idx = start_idx + window_samples
                    if end_idx <= crop_data.shape[-1]:
                        physio_data.append(crop_data[:, start_idx:(end_idx + offset)])
                    else:
                        print(f"Warning: Window at {start} s exceeds data boundaries")
                
                physio_data = np.array(physio_data)
                print(f"Chunked MEG shape: {physio_data.shape}")
                print(f"Each chunk duration: {physio_data.shape[-1]/sfreq:.3f}s")
                
                np.save(os.path.join(save_dir, f"run-{run}.npy"), physio_data)

                del raw, raw_resampled, raw_cropped, crop_data, physio_data
                gc.collect()
        
    if args.word:
        logging.info("--- Chunking Word-Level MEG Data ---")
        DURATION = DURATION_WORD
        trans_dir = os.path.join(SAVE_ROOT, "word_transcription")
        for sub in ["sub-01","sub-02","sub-03","sub-04","sub-05",
                    "sub-06","sub-07","sub-10","sub-11","sub-12"]:
            save_dir = os.path.join(save_root, "word_chunkphysio", sub)
            os.makedirs(save_dir, exist_ok=True)
            for run in range(1, 61):
                logging.info(f"Processing Run {run} for Subject '{sub}'... ")
                bids_path = BIDSPath(subject=sub.split("-")[-1],
                                    task="RDR", run=f"{run}", suffix="meg", datatype="MEG",
                                    root=PREPROC_ROOT, extension=".fif", check=False)
                print(bids_path)

                raw = mne.io.read_raw_fif(bids_path, preload=True)
                original_sfreq = raw.info["sfreq"]
                new_sfreq = original_sfreq / DECIM
                raw_resampled = raw.copy().resample(new_sfreq)
                
                events = mne.find_events(raw_resampled, shortest_event=1)
                sfreq = raw_resampled.info["sfreq"]
                window_samples = int(DURATION * sfreq)
                offset = int(MAX_TIMELAG * new_sfreq)
                
                print(f"Original sfreq: {original_sfreq}, New sfreq: {sfreq}")
                print(f"Window samples for {int(DURATION * 1000)} ms: {window_samples}")
                print(f"Offset time lag samples for {int(MAX_TIMELAG * 1000)} ms: {offset}")

                start_time = (events[0, 0] - raw_resampled.first_samp) / sfreq
                end_time = (events[1, 0] - raw_resampled.first_samp) / sfreq
                
                raw_cropped = raw_resampled.copy().crop(tmin=start_time, tmax=end_time + DURATION + MAX_TIMELAG).pick(picks="meg")
                crop_data = raw_cropped.get_data()
                print(f"Cropped data shape: {crop_data.shape}")
                
                duration = end_time - start_time + DURATION + MAX_TIMELAG
                timevec = np.linspace(0, duration, crop_data.shape[-1])

                physio_data = list()
                trans_file = os.path.join(trans_dir, f"story_{run}.csv")
                start_times = pd.read_csv(trans_file)["start"].tolist()
                for start in start_times:
                    start_idx = np.argmin(np.abs(timevec - start))
                    end_idx = start_idx + window_samples
                    if end_idx <= crop_data.shape[-1]:
                        physio_data.append(crop_data[:, start_idx:(end_idx + offset)])
                    else:
                        print(f"Warning: Window at {start} s exceeds data boundaries")
                
                physio_data = np.array(physio_data)
                print(f"Chunked MEG shape: {physio_data.shape}")
                print(f"Each chunk duration: {physio_data.shape[-1]/sfreq:.3f}s")
                
                np.save(os.path.join(save_dir, f"run-{run}.npy"), physio_data)

                del raw, raw_resampled, raw_cropped, crop_data, physio_data
                gc.collect()

    if not args.char and not args.word:
        parser.print_help()