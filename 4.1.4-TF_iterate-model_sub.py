import os
import torch
import logging
import argparse
import numpy as np

from mne.filter import resample
from torch.utils.data import TensorDataset, DataLoader, random_split

from func.utils import check_memory, set_seed
from func.configs import WHISPER_SEG_LEN, RESAMPLE_SFREQ, DECIM, DURATION_CHAR, DURATION_WORD, model_paths
from func.dnn import TRFModel
from func.dnn_train_eval import train_model, evaluate_model

# Directories
DATA_ROOT = "./data/saved"
RESULTS_ROOT = "./results/TF_iterate-model_sub"

# Global Parameters
SEED = 1016
TRAIN_PROPOTION = 0.7
VALID_PROPOTION = 0.1
TEST_PROPOTION = 0.2
FIXED_CONTEXT_LEN = 1000
FIXED_TIMELAG = 300
FEATURE = "speech"
SFREQ = RESAMPLE_SFREQ / DECIM # 200 Hz
WHISPER_SFREQ = 1000 / WHISPER_SEG_LEN # 50 Hz
DTYPE = np.float32
SCALE = 1e10
TF_PARAMS = {
    "d_model" : 512,
    "using_tf" : True,
    "n_blocks" : 4
}
TRAIN_PARAMS = {
    "num_epochs" : 200,
    "learning_rate" : 0.01,
    "weight_decay" : 1e-4,
    "patience" : 10
}
BATCH_SIZE = 256


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

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    if args.char:
        logging.info("--- DNN-Based TRF for Char-Level MEG Data ---")
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
            train_size = int(N_SAMPLES * TRAIN_PROPOTION)
            val_size = int(N_SAMPLES * VALID_PROPOTION)
            test_size = N_SAMPLES - train_size - val_size

            start_idx = FIXED_TIMELAG // WHISPER_SEG_LEN
            end_idx = start_idx + int(DURARION * 1000) // WHISPER_SEG_LEN

            physio_data = physio_data[:, start_idx:end_idx, :]
            physio_data = torch.from_numpy(physio_data).float()

            for model_type in model_paths.keys():
                logging.info(f"Conducting mTRF for: Subject '{sub}', Model '{model_type}', Context {FIXED_CONTEXT_LEN}ms, Feature '{FEATURE}', TimeLag {FIXED_TIMELAG}ms...")
                embeds_dir = os.path.join(DATA_ROOT, f"whisper_features/char-level/{model_type}")
                embeds = list()
                for run in range(1, 61):
                    embeds.append(np.load(os.path.join(embeds_dir, f"whisper_{FEATURE}_context-{FIXED_CONTEXT_LEN}_story_{run}.npy")))
                embeds = np.concatenate(embeds, axis=0).astype(DTYPE)

                _, _, EMBED_DIM = embeds.shape

                if embeds.shape[0] != N_SAMPLES: 
                    raise ValueError(f"Inconsistent samples for embeddings and MEG data. Got: {embeds.shape[0]} and {N_SAMPLES}.")
                
                embeds = torch.from_numpy(embeds).float()
                dataset = TensorDataset(embeds, physio_data)
                
                train_dataset, val_dataset, test_dataset = random_split(
                    dataset, [train_size, val_size, test_size],
                    generator= torch.Generator().manual_seed(SEED)
                )

                train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_JOBS, pin_memory=True)
                val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=N_JOBS)
                test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=N_JOBS)

                model = TRFModel(n_signals=N_VERTICES, n_dims=EMBED_DIM, device=device, **TF_PARAMS)
                train_model(model, train_loader, val_loader, device=device,
                            save_dir=results_dir, fname=f"whisper-{model_type}_tf.pt", **TRAIN_PARAMS)
                
                r = evaluate_model(model, test_loader, device=device)
                logging.info(f"Max r-value: {r.max():.3f}, Mean r-value: {r.mean():.3f}")
                np.save(os.path.join(results_dir, f"whisper-{model_type}_r-obs.npy"), r)
    
    if args.word:
        logging.info("--- DNN-Based TRF for Word-Level MEG Data ---")
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
            train_size = int(N_SAMPLES * TRAIN_PROPOTION)
            val_size = int(N_SAMPLES * VALID_PROPOTION)
            test_size = N_SAMPLES - train_size - val_size

            start_idx = FIXED_TIMELAG // WHISPER_SEG_LEN
            end_idx = start_idx + int(DURARION * 1000) // WHISPER_SEG_LEN

            physio_data = physio_data[:, start_idx:end_idx, :]
            physio_data = torch.from_numpy(physio_data).float()

            for model_type in model_paths.keys():
                logging.info(f"Conducting mTRF for: Subject '{sub}', Model '{model_type}', Context {FIXED_CONTEXT_LEN}ms, Feature '{FEATURE}', TimeLag {FIXED_TIMELAG}ms...")
                embeds_dir = os.path.join(DATA_ROOT, f"whisper_features/word-level/{model_type}")
                embeds = list()
                for run in range(1, 61):
                    embeds.append(np.load(os.path.join(embeds_dir, f"whisper_{FEATURE}_context-{FIXED_CONTEXT_LEN}_story_{run}.npy")))
                embeds = np.concatenate(embeds, axis=0).astype(DTYPE)

                _, _, EMBED_DIM = embeds.shape

                if embeds.shape[0] != N_SAMPLES: 
                    raise ValueError(f"Inconsistent samples for embeddings and MEG data. Got: {embeds.shape[0]} and {N_SAMPLES}.")
                
                embeds = torch.from_numpy(embeds).float()
                dataset = TensorDataset(embeds, physio_data)
                
                train_dataset, val_dataset, test_dataset = random_split(
                    dataset, [train_size, val_size, test_size],
                    generator= torch.Generator().manual_seed(SEED)
                )

                train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_JOBS, pin_memory=True)
                val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=N_JOBS)
                test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=N_JOBS)

                model = TRFModel(n_signals=N_VERTICES, n_dims=EMBED_DIM, device=device, **TF_PARAMS)
                train_model(model, train_loader, val_loader, device=device,
                            save_dir=results_dir, fname=f"whisper-{model_type}_tf.pt", **TRAIN_PARAMS)
                
                r = evaluate_model(model, test_loader, device=device)
                logging.info(f"Max r-value: {r.max():.3f}, Mean r-value: {r.mean():.3f}")
                np.save(os.path.join(results_dir, f"whisper-{model_type}_r-obs.npy"), r)
    
    if not args.char and not args.word:
        parser.print_help()