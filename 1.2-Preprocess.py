import os
import math
import argparse
import logging
import pandas as pd
import numpy as np
import mne

from mne.io import read_raw_fif

from func.configs import RESAMPLE_SFREQ
from func.configs import reject_criteria as reject


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    parser = argparse.ArgumentParser(description="The script to perform pre-processing procedures for OpenNeuro datasets.")

    parser.add_argument("--n_jobs", default=32, help="Number of jobs to run in parallel.")
    
    args = parser.parse_args()

    N_JOBS = args.n_jobs
    
    target_root = "./data/bids"

    dataset = "SMN4Lang"
    logging.info(f"--- Processing Dataset: {dataset} ---")
    bids_root = os.path.join(target_root, dataset)
    
    ## This dataset provides only task data, processed by MAXFilter, SSS and preprocessing procedures.
    ## Here, we used the time period before stimulus to compute noise covariance matrix.
    ## However, each time period is too short (around 10s) to estimate noise covariance matrix (at least 47s needed).
    ## So, we would combine these time periods together for estimation.
    line_freq = 50
    bids_root = os.path.join(bids_root, "derivatives/preprocessed_data")
    for sub in os.listdir(bids_root):
        raws = list()
        for run in range(1, 61):
            df = pd.read_csv(os.path.join(bids_root, f"{sub}/MEG/{sub}_task-RDR_run-{run}_events.tsv"), sep="\t")
            onset = df["onset"].values[np.where(df["trial_type"].values == "Beg")[0]]
            onset = math.floor(onset[0]) - 3
            raw = read_raw_fif(
                os.path.join(bids_root, f"{sub}/MEG/{sub}_task-RDR_run-{run}_meg.fif"),
                preload=True
            )
            raw = raw.resample(sfreq=RESAMPLE_SFREQ, npad="auto", n_jobs=N_JOBS)
            raw.load_data().notch_filter(np.arange(line_freq, line_freq * 7 + 1, line_freq))
            raw.save(
                os.path.join(bids_root, f"{sub}/MEG/{sub}_task-RDR_run-{run}_meg.fif"),
                overwrite=True
            )
            raw = raw.crop(tmin=0., tmax=onset)
            raws.append(raw)
        raws = mne.concatenate_raws(raws, on_mismatch="ignore")
        noise_cov = mne.compute_raw_covariance(raws, 
                                                tmin=0, tmax=None,
                                                picks="meg", reject=reject,
                                                method="auto", cv=5, rank="info")
        cov_root = os.path.join(bids_root, f"{sub}/MEG/noise_cov")
        if not os.path.exists(cov_root):
            os.mkdir(cov_root)
        mne.write_cov(os.path.join(cov_root, "noise-cov.fif"), noise_cov, overwrite=True)