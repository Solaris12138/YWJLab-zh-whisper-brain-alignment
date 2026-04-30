import os
import mne

from mne.io import read_info
from mne.minimum_norm import make_inverse_operator, write_inverse_operator


if __name__ == "__main__":

    current_dir = os.getcwd()
    subjects_dir = os.path.join(current_dir, "data/bids/SMN4Lang/freesurfer")
    root = "./data/bids/SMN4Lang/derivatives/preprocessed_data"

    for subject in ["sub-01","sub-02","sub-03","sub-04","sub-05","sub-06",
                    "sub-07","sub-08","sub-09","sub-10","sub-11","sub-12"]:

        fwd_root = os.path.join(root, f"{subject}/MEG/fwd")
        cov_root = os.path.join(root, f"{subject}/MEG/noise_cov")
        inv_root = os.path.join(root, f"{subject}/MEG/inv")
        if not os.path.exists(cov_root):
            os.mkdir(cov_root)
        if not os.path.exists(inv_root):
            os.mkdir(inv_root)

        noise_cov = mne.read_cov(os.path.join(cov_root, "noise-cov.fif"))
        
        ## Compute inverse operator
        for run in range(1, 61):
            print(f"Processing Subject: {subject}, Run {run}")

            fname = os.path.join(root, f"{subject}/MEG/{subject}_task-RDR_run-{run}_meg.fif")

            info = read_info(fname)

            basename = os.path.basename(fname)

            fwd = mne.read_forward_solution(
                os.path.join(
                    fwd_root,
                    basename.replace("_meg.fif", "-fwd.fif")
                )
            )

            inv = make_inverse_operator(info, fwd, noise_cov, loose=0.2, depth=0.8, fixed="auto", rank="info")

            write_inverse_operator(os.path.join(inv_root, basename.replace("_meg.fif", "-inv.fif")),
                                   inv, overwrite=True)