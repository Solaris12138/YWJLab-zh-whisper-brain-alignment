import os
import gc
import logging
import argparse
import numpy as np

from mtrf.model import TRF
from mtrf.stats import pearsonr
from mne.filter import resample
from sklearn.model_selection import train_test_split

from func.utils import check_memory, set_seed
from func.configs import WHISPER_SEG_LEN, RESAMPLE_SFREQ, DECIM, DURATION_CHAR, DURATION_WORD, model_paths

# Directories
DATA_ROOT = "./data/saved"
RESULTS_ROOT = "./results/mTRF_iterate-model_sub"

# Global Parameters
SEED = 1016
DATA_PROPOTION = 0.7
FIXED_CONTEXT_LEN = 1000
FIXED_TIMELAG = 300
FEATURE = "speech"
SFREQ = RESAMPLE_SFREQ / DECIM # 200 Hz
WHISPER_SFREQ = 1000 / WHISPER_SEG_LEN # 50 Hz
TRF_PARAMS = {
    "direction" : 1,
    "kind" : "single",
    "method" : "ridge",
    "preload" : True,
    "metric" : pearsonr
}
TRAIN_PARAMS = {
    "fs" : WHISPER_SFREQ,
    "tmin" : 0,
    "tmax" : 0,
    "bands" : None,
    "k" : 3,
    "average" : True,
    "seed" : SEED,
    "reg_per_y_channel" : False
}
REG = np.logspace(-1, 5, 10)
PERM_PARAMS = {
    "fs" : WHISPER_SFREQ,
    "tmin" : 0,
    "tmax" : 0,
    "n_permute" : 1000,
    "k" : 3,
    "seed" : SEED,
    "average" : False
}
DTYPE = np.float32
SCALE = 1e10


if __name__ == "__main__":
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )

    parser = argparse.ArgumentParser(description="A script to perform single-subject mTRF.")
    parser.add_argument("--char", help="Using char-level MEG data.", action="store_true")
    parser.add_argument("--word", help="Using word-level MEG data.", action="store_true")
    parser.add_argument("--n_jobs", default=32, help="Number of jobs to run in parallel.")
    args = parser.parse_args()

    N_JOBS = args.n_jobs

    check_memory()
    set_seed(SEED)

    if args.char:
        logging.info("--- mTRF for Char-Level MEG Data ---")
        DURARION = DURATION_CHAR
        for sub in ["sub-01","sub-02","sub-03","sub-04","sub-05",
                    "sub-06","sub-07","sub-10","sub-11","sub-12"]:
            results_dir = os.path.join(RESULTS_ROOT, "char-level", f"{sub}")
            os.makedirs(results_dir, exist_ok=True)

            physio_data_dir = os.path.join(DATA_ROOT, f"ChunkedPhysioData/source_level/char_chunkphysio/{sub}")
            physio_data = list()
            for run in range(1, 51):
                physio_data.append(np.load(os.path.join(physio_data_dir, f"run-{run}.npy")))
            physio_data = np.concatenate(physio_data, axis=0)
            physio_data = resample(physio_data, down=SFREQ//WHISPER_SFREQ, 
                                   axis=-1, n_jobs=N_JOBS)
            physio_data = np.transpose(physio_data, (0, 2, 1))
            physio_data = (physio_data * SCALE).astype(DTYPE)

            N_SAMPLES = physio_data.shape[0]
            
            train_idx, test_idx = train_test_split(np.arange(N_SAMPLES), test_size=1-DATA_PROPOTION, random_state=SEED)
            train_pyshio_data, test_pysio_data = physio_data[train_idx], physio_data[test_idx]

            start_idx = FIXED_TIMELAG // WHISPER_SEG_LEN
            end_idx = start_idx + int(DURARION * 1000) // WHISPER_SEG_LEN

            train_pyshio_data_lag = [train_pyshio_data[i, start_idx:end_idx, :] for i in range(train_pyshio_data.shape[0])]
            test_pysio_data_lag = [test_pysio_data[i, start_idx:end_idx, :] for i in range(test_pysio_data.shape[0])]

            del physio_data, train_pyshio_data, test_pysio_data
            gc.collect()

            for model_type in model_paths.keys():
                logging.info(f"Conducting mTRF for: Subject '{sub}', Model '{model_type}', Context {FIXED_CONTEXT_LEN}ms, Feature '{FEATURE}', TimeLag {FIXED_TIMELAG}ms...")
                embeds_dir = os.path.join(DATA_ROOT, f"whisper_features/char-level/{model_type}")
                embeds = list()
                for run in range(1, 51):
                    embeds.append(np.load(os.path.join(embeds_dir, f"whisper_{FEATURE}_context-{FIXED_CONTEXT_LEN}_story_{run}.npy")))
                embeds = np.concatenate(embeds, axis=0).astype(DTYPE)

                if embeds.shape[0] != N_SAMPLES: 
                    raise ValueError(f"Inconsistent samples for embeddings and MEG data. Got: {embeds.shape[0]} and {N_SAMPLES}.")
                train_embeds, test_embeds = embeds[train_idx], embeds[test_idx]

                del embeds
                gc.collect()

                train_embeds = [train_embeds[i] for i in range(train_embeds.shape[0])]
                test_embeds = [test_embeds[i] for i in range(test_embeds.shape[0])]

                trf = TRF(**TRF_PARAMS)
                trf.train(train_embeds, train_pyshio_data_lag, regularization=REG, **TRAIN_PARAMS)

                recon_pysio_data_lag, r = trf.predict(test_embeds, test_pysio_data_lag, lag=None, average=False)
                logging.info(f"Max r-value: {r.max():.3f}, Mean r-value: {r.mean():.3f}")
                np.save(os.path.join(results_dir, f"whisper-{model_type}_r-obs.npy"), r)
    
    if args.word:
        logging.info("--- mTRF for Word-Level MEG Data ---")
        DURARION = DURATION_WORD
        for sub in ["sub-01","sub-02","sub-03","sub-04","sub-05",
                    "sub-06","sub-07","sub-10","sub-11","sub-12"]:
            results_dir = os.path.join(RESULTS_ROOT, "word-level", f"{sub}")
            os.makedirs(results_dir, exist_ok=True)

            physio_data_dir = os.path.join(DATA_ROOT, f"ChunkedPhysioData/source_level/word_chunkphysio/{sub}")
            physio_data = list()
            for run in range(1, 51):
                physio_data.append(np.load(os.path.join(physio_data_dir, f"run-{run}.npy")))
            physio_data = np.concatenate(physio_data, axis=0)
            physio_data = resample(physio_data, down=SFREQ//WHISPER_SFREQ, 
                                   axis=-1, n_jobs=N_JOBS)
            physio_data = np.transpose(physio_data, (0, 2, 1))
            physio_data = (physio_data * SCALE).astype(DTYPE)

            N_SAMPLES = physio_data.shape[0]
            
            train_idx, test_idx = train_test_split(np.arange(N_SAMPLES), test_size=1-DATA_PROPOTION, random_state=SEED)
            train_pyshio_data, test_pysio_data = physio_data[train_idx], physio_data[test_idx]

            start_idx = FIXED_TIMELAG // WHISPER_SEG_LEN
            end_idx = start_idx + int(DURARION * 1000) // WHISPER_SEG_LEN

            train_pyshio_data_lag = [train_pyshio_data[i, start_idx:end_idx, :] for i in range(train_pyshio_data.shape[0])]
            test_pysio_data_lag = [test_pysio_data[i, start_idx:end_idx, :] for i in range(test_pysio_data.shape[0])]

            del physio_data, train_pyshio_data, test_pysio_data
            gc.collect()

            for model_type in model_paths.keys():
                logging.info(f"Conducting mTRF for: Subject '{sub}', Model '{model_type}', Context {FIXED_CONTEXT_LEN}ms, Feature '{FEATURE}', TimeLag {FIXED_TIMELAG}ms...")
                embeds_dir = os.path.join(DATA_ROOT, f"whisper_features/word-level/{model_type}")
                embeds = list()
                for run in range(1, 51):
                    embeds.append(np.load(os.path.join(embeds_dir, f"whisper_{FEATURE}_context-{FIXED_CONTEXT_LEN}_story_{run}.npy")))
                embeds = np.concatenate(embeds, axis=0).astype(DTYPE)

                if embeds.shape[0] != N_SAMPLES: 
                    raise ValueError(f"Inconsistent samples for embeddings and MEG data. Got: {embeds.shape[0]} and {N_SAMPLES}.")
                train_embeds, test_embeds = embeds[train_idx], embeds[test_idx]

                del embeds
                gc.collect()

                train_embeds = [train_embeds[i] for i in range(train_embeds.shape[0])]
                test_embeds = [test_embeds[i] for i in range(test_embeds.shape[0])]

                trf = TRF(**TRF_PARAMS)
                trf.train(train_embeds, train_pyshio_data_lag, regularization=REG, **TRAIN_PARAMS)

                recon_pysio_data_lag, r = trf.predict(test_embeds, test_pysio_data_lag, lag=None, average=False)
                logging.info(f"Max r-value: {r.max():.3f}, Mean r-value: {r.mean():.3f}")
                np.save(os.path.join(results_dir, f"whisper-{model_type}_r-obs.npy"), r)

    if not args.char and not args.word:
        parser.print_help()             