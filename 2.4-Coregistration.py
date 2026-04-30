import os
import numpy as np
import mne

from mne.coreg import Coregistration
from mne.io import read_info


if __name__ == "__main__":

    root = "./data/bids/SMN4Lang/derivatives/preprocessed_data"
    subjects_dir = "./data/bids/SMN4Lang/freesurfer"

    for subject in ["sub-01","sub-02","sub-03","sub-04","sub-05","sub-06",
                    "sub-07","sub-08","sub-09","sub-10","sub-11","sub-12"]:
        
        fiducials = "auto"
        
        for run in range(1, 61):
            print(f"Processing Subject: {subject}, Run {run}")
            info = read_info(
                os.path.join(
                    root,
                    f"{subject}/MEG/{subject}_task-RDR_run-{run}_meg.fif"
                )
            )

            coreg = Coregistration(info, subject, subjects_dir, fiducials=fiducials)
            coreg.fit_fiducials(verbose=True)
            coreg.fit_icp(n_iterations=100, nasion_weight=5., verbose=True)
            coreg.omit_head_shape_points(distance=5. / 1000)
            coreg.fit_icp(n_iterations=50, nasion_weight=10., verbose=True)
            dists = coreg.compute_dig_mri_distances() * 1e3
            print(
                f"Distance between HSP and MRI (mean/min/max):\n{np.mean(dists):.2f} mm "
                f"/ {np.min(dists):.2f} mm / {np.max(dists):.2f} mm"
            )

            trans_root = os.path.join(root, f"{subject}/MEG/trans")
            if not os.path.exists(trans_root):
                os.mkdir(trans_root)
            
            mne.write_trans(os.path.join(trans_root, f"{subject}_task-RDR_run-{run}-trans.fif"), 
                            coreg.trans, overwrite=True)