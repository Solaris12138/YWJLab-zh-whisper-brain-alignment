import os
import mne
import pickle
import numpy as np
import argparse
import logging

from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator

from func.configs import ATLAS, AUDLANG_NET
from func.utils import combine_labels


# Directories
BIDS_ROOT = "./data/bids/SMN4Lang"
PREPROC_ROOT = os.path.join(BIDS_ROOT, "derivatives/preprocessed_data")
SAVE_ROOT = "./data/saved"
SUBJECTS_DIR = os.path.join(BIDS_ROOT, "freesurfer")

# Global Parameters
SNR = 3.0
INV_PARAMS = {
    "lambda2" : 1.0 / SNR ** 2,
    "method" : "eLORETA",
    "pick_ori" : "normal",
    "return_generator" : True,
    "verbose" : "error"
}
SUBJECT_TO = "fsaverage"
SPACING = "ico3"


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )

    parser = argparse.ArgumentParser(description="A script to transform MEG data from sensor-level to source-level.")
    parser.add_argument("--char", help="Using char-level MEG data.", action="store_true")
    parser.add_argument("--word", help="Using word-level MEG data.", action="store_true")
    args = parser.parse_args()

    label = combine_labels(SUBJECT_TO, SUBJECTS_DIR, ATLAS, AUDLANG_NET)

    save_root = os.path.join(SAVE_ROOT, "ChunkedPhysioData/source_level")

    info = pickle.load(open("neuromag_info.pkl", "rb"))

    if args.char:
        logging.info("--- Source Estimation for Char-Level MEG Data ---")
        for sub in ["sub-01","sub-02","sub-03","sub-04","sub-05",
                    "sub-06","sub-07","sub-10","sub-11","sub-12"]:
            save_dir = os.path.join(save_root, "char_chunkphysio", sub)
            os.makedirs(save_dir, exist_ok=True)
            for run in range(1, 61):
                logging.info(f"Processing Run {run} for Subject '{sub}'... ")

                physio_data = np.load(
                    os.path.join(
                        SAVE_ROOT,
                        "ChunkedPhysioData/sensor_level",
                        "char_chunkphysio",
                        sub,
                        f"run-{run}.npy"
                    )
                )
                epoch_run = mne.EpochsArray(physio_data, info)

                inv = read_inverse_operator(os.path.join(PREPROC_ROOT, f"{sub}/MEG/inv/{sub}_task-RDR_run-{run}-inv.fif"))
                morph = mne.read_source_morph(os.path.join(SUBJECTS_DIR, f"{sub}/bem/{sub}2{SUBJECT_TO}-{SPACING}-morph.h5"))

                stcs = apply_inverse_epochs(epoch_run, inv, **INV_PARAMS)
                stcs_fs = np.array([morph.apply(stc).in_label(label).data for stc in stcs])

                np.save(os.path.join(save_dir, f"run-{run}.npy"), stcs_fs)
                print(stcs_fs.min())
                print(f"MEG data shape: {stcs_fs.shape}")
    
    if args.word:
        logging.info("--- Source Estimation for Word-Level MEG Data ---")
        for sub in ["sub-01","sub-02","sub-03","sub-04","sub-05",
                    "sub-06","sub-07","sub-10","sub-11","sub-12"]:
            save_dir = os.path.join(save_root, "word_chunkphysio", sub)
            os.makedirs(save_dir, exist_ok=True)
            for run in range(1, 61):
                logging.info(f"Processing Run {run} for Subject '{sub}'... ")

                physio_data = np.load(
                    os.path.join(
                        SAVE_ROOT,
                        "ChunkedPhysioData/sensor_level",
                        "word_chunkphysio",
                        sub,
                        f"run-{run}.npy"
                    )
                )
                epoch_run = mne.EpochsArray(physio_data, info)

                inv = read_inverse_operator(os.path.join(PREPROC_ROOT, f"{sub}/MEG/inv/{sub}_task-RDR_run-{run}-inv.fif"))
                morph = mne.read_source_morph(os.path.join(SUBJECTS_DIR, f"{sub}/bem/{sub}2{SUBJECT_TO}-{SPACING}-morph.h5"))

                stcs = apply_inverse_epochs(epoch_run, inv, **INV_PARAMS)
                stcs_fs = np.array([morph.apply(stc).in_label(label).data for stc in stcs])

                np.save(os.path.join(save_dir, f"run-{run}.npy"), stcs_fs)
                print(stcs_fs.min())
                print(f"MEG data shape: {stcs_fs.shape}")
    
    if not args.char and not args.word:
        parser.print_help()