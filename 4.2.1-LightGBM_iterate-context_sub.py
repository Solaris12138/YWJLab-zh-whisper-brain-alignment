import os
import gc
import logging
import argparse
import numpy as np

from mne.filter import resample
from sklearn.model_selection import train_test_split

from func.utils import check_memory, set_seed
from func.configs import WHISPER_SEG_LEN, RESAMPLE_SFREQ, DECIM, DURATION_CHAR, DURATION_WORD, CONTEXT_LEN
from func.lgbm import lgbm_vertices_parallel

# Directories
DATA_ROOT = "./data/saved"
RESULTS_ROOT = "./results/LightGBM_iterate-context_sub"

# Global Parameters
SEED = 1016
DATA_PROPOTION = 0.8
VALID_PROPOTION = 0.2
FIXED_MODEL = "zh-ft"
FIXED_TIMELAG = 300
FEATURES = ["acoustics", "speech"]
SFREQ = RESAMPLE_SFREQ / DECIM # 200 Hz
WHISPER_SFREQ = 1000 / WHISPER_SEG_LEN # 50 Hz
DTYPE = np.float32
SCALE = 1e10
LGBM_PARAMS = {
    "boosting_type" : "gbdt",
    "n_estimators" : 100,
    "learning_rate" : 0.05,
    "reg_lambda" : 1e-4,
    "early_stopping" : 10,
    "random_state" : SEED
}


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
        logging.info("--- LGBM-Based TRF for Char-Level MEG Data ---")
        DURARION = DURATION_CHAR
        for sub in ["sub-01","sub-02","sub-03","sub-04","sub-05",
                    "sub-06","sub-07","sub-10","sub-11","sub-12"]:
            results_dir = os.path.join(RESULTS_ROOT, "char-level", f"{sub}")
            os.makedirs(results_dir, exist_ok=True)

            physio_data_dir = os.path.join(DATA_ROOT, f"ChunkedPhysioData/source_level/char_chunkphysio/{sub}")
            physio_data = list()
            for run in range(1, 61):
                physio_data.append(np.load(os.path.join(physio_data_dir, f"run-{run}.npy")))
            physio_data = np.concatenate(physio_data, axis=0)
            physio_data = resample(physio_data, down=SFREQ//WHISPER_SFREQ, 
                                   axis=-1, n_jobs=N_JOBS)
            physio_data = np.transpose(physio_data, (0, 2, 1))
            physio_data = (physio_data * SCALE).astype(DTYPE)

            N_SAMPLES, _, N_VERTICES = physio_data.shape
            
            train_idx, test_idx = train_test_split(np.arange(N_SAMPLES), test_size=1-DATA_PROPOTION, random_state=SEED)
            y_train, y_test = physio_data[train_idx], physio_data[test_idx]

            start_idx = FIXED_TIMELAG // WHISPER_SEG_LEN
            end_idx = start_idx + int(DURARION * 1000) // WHISPER_SEG_LEN

            y_train = y_train[:, start_idx:end_idx, :]
            y_test = y_test[:, start_idx:end_idx, :]

            del physio_data
            gc.collect()

            y_train = y_train.reshape(-1, N_VERTICES)
            y_test = y_test.reshape(-1, N_VERTICES)

            valid_boundary = int(y_train.shape[0] * VALID_PROPOTION)
            y_train , y_val = y_train[valid_boundary:], y_train[:valid_boundary]

            for feature in FEATURES:
                for context_len in CONTEXT_LEN:
                    logging.info(f"Conducting mTRF for: Subject '{sub}', Model '{FIXED_MODEL}', Context {context_len}ms, Feature '{feature}', TimeLag {FIXED_TIMELAG}ms...")
                    embeds_dir = os.path.join(DATA_ROOT, f"whisper_features/char-level/{FIXED_MODEL}")
                    embeds = list()
                    for run in range(1, 61):
                        embeds.append(np.load(os.path.join(embeds_dir, f"whisper_{feature}_context-{context_len}_story_{run}.npy")))
                    embeds = np.concatenate(embeds, axis=0).astype(DTYPE)

                    _, _, EMBED_DIM = embeds.shape

                    if embeds.shape[0] != N_SAMPLES: 
                        raise ValueError(f"Inconsistent samples for embeddings and MEG data. Got: {embeds.shape[0]} and {N_SAMPLES}.")
                    X_train, X_test = embeds[train_idx], embeds[test_idx]

                    del embeds
                    gc.collect()

                    X_train = X_train.reshape(-1, EMBED_DIM)
                    X_test = X_test.reshape(-1, EMBED_DIM)
                    
                    X_train , X_val = X_train[valid_boundary:], X_train[:valid_boundary]

                    r = lgbm_vertices_parallel(X_train, y_train, X_val, y_val, X_test, y_test, n_jobs=N_JOBS, **LGBM_PARAMS)
                    logging.info(f"Max r-value: {r.max():.3f}, Mean r-value: {r.mean():.3f}")
                    np.save(os.path.join(results_dir, f"whisper-{feature}_context-{context_len}_r-obs.npy"), r)
    
    if args.word:
        logging.info("--- LGBM-Based TRF for Word-Level MEG Data ---")
        DURARION = DURATION_WORD
        for sub in ["sub-01","sub-02","sub-03","sub-04","sub-05",
                    "sub-06","sub-07","sub-10","sub-11","sub-12"]:
            results_dir = os.path.join(RESULTS_ROOT, "word-level", f"{sub}")
            os.makedirs(results_dir, exist_ok=True)

            physio_data_dir = os.path.join(DATA_ROOT, f"ChunkedPhysioData/source_level/word_chunkphysio/{sub}")
            physio_data = list()
            for run in range(1, 61):
                physio_data.append(np.load(os.path.join(physio_data_dir, f"run-{run}.npy")))
            physio_data = np.concatenate(physio_data, axis=0)
            physio_data = resample(physio_data, down=SFREQ//WHISPER_SFREQ, 
                                   axis=-1, n_jobs=N_JOBS)
            physio_data = np.transpose(physio_data, (0, 2, 1))
            physio_data = (physio_data * SCALE).astype(DTYPE)

            N_SAMPLES, _, N_VERTICES = physio_data.shape
            
            train_idx, test_idx = train_test_split(np.arange(N_SAMPLES), test_size=1-DATA_PROPOTION, random_state=SEED)
            y_train, y_test = physio_data[train_idx], physio_data[test_idx]

            start_idx = FIXED_TIMELAG // WHISPER_SEG_LEN
            end_idx = start_idx + int(DURARION * 1000) // WHISPER_SEG_LEN

            y_train = y_train[:, start_idx:end_idx, :]
            y_test = y_test[:, start_idx:end_idx, :]

            del physio_data
            gc.collect()

            y_train = y_train.reshape(-1, N_VERTICES)
            y_test = y_test.reshape(-1, N_VERTICES)

            valid_boundary = int(y_train.shape[0] * VALID_PROPOTION)
            y_train , y_val = y_train[valid_boundary:], y_train[:valid_boundary]

            for feature in FEATURES:
                for context_len in CONTEXT_LEN:
                    logging.info(f"Conducting mTRF for: Subject '{sub}', Model '{FIXED_MODEL}', Context {context_len}ms, Feature '{feature}', TimeLag {FIXED_TIMELAG}ms...")
                    embeds_dir = os.path.join(DATA_ROOT, f"whisper_features/word-level/{FIXED_MODEL}")
                    embeds = list()
                    for run in range(1, 61):
                        embeds.append(np.load(os.path.join(embeds_dir, f"whisper_{feature}_context-{context_len}_story_{run}.npy")))
                    embeds = np.concatenate(embeds, axis=0).astype(DTYPE)

                    _, _, EMBED_DIM = embeds.shape

                    if embeds.shape[0] != N_SAMPLES: 
                        raise ValueError(f"Inconsistent samples for embeddings and MEG data. Got: {embeds.shape[0]} and {N_SAMPLES}.")
                    X_train, X_test = embeds[train_idx], embeds[test_idx]

                    del embeds
                    gc.collect()

                    X_train = X_train.reshape(-1, EMBED_DIM)
                    X_test = X_test.reshape(-1, EMBED_DIM)
                    
                    X_train , X_val = X_train[valid_boundary:], X_train[:valid_boundary]

                    r = lgbm_vertices_parallel(X_train, y_train, X_val, y_val, X_test, y_test, n_jobs=N_JOBS, **LGBM_PARAMS)
                    logging.info(f"Max r-value: {r.max():.3f}, Mean r-value: {r.mean():.3f}")
                    np.save(os.path.join(results_dir, f"whisper-{feature}_context-{context_len}_r-obs.npy"), r)
    
    if not args.char and not args.word:
        parser.print_help()