import os
import mne

from mne.io import read_info


if __name__ == "__main__":
    
    bem_ico = 5
    
    current_dir = os.getcwd()
    subjects_dir = os.path.join(current_dir, "data/bids/SMN4Lang/freesurfer")
    root = "./data/bids/SMN4Lang/derivatives/preprocessed_data"
    
    for subject in ["sub-01","sub-02","sub-03","sub-04","sub-05","sub-06",
                    "sub-07","sub-08","sub-09","sub-10","sub-11","sub-12"]:
        
        trans_root = os.path.join(root, f"{subject}/MEG/trans")
        fwd_root = os.path.join(root, f"{subject}/MEG/fwd")
        if not os.path.exists(fwd_root):
            os.mkdir(fwd_root)
        
        bem_sol_fname = os.path.join(subjects_dir, subject, "bem",
                                     f"{subject}-ico{bem_ico}-bem-sol.fif")
        src_fname = os.path.join(subjects_dir, subject, "bem",
                                 f"{subject}-ico{bem_ico}-src.fif")
        
        bem = mne.read_bem_solution(bem_sol_fname)
        src = mne.read_source_spaces(src_fname)
        
        for run in range(1, 61):
            print(f"Processing Subject: {subject}, Run {run}")
            info = read_info(
                os.path.join(
                    root,
                    f"{subject}/MEG/{subject}_task-RDR_run-{run}_meg.fif"
                )
            )

            trans = mne.read_trans(
                os.path.join(
                    trans_root, 
                    f"{subject}_task-RDR_run-{run}-trans.fif"
                )
            )

            fwd = mne.make_forward_solution(info, trans, src, bem,
                                            meg=True, eeg=False, mindist=5.0, 
                                            n_jobs=None, verbose=True,)

            mne.write_forward_solution(os.path.join(fwd_root, f"{subject}_task-RDR_run-{run}-fwd.fif"), 
                                       fwd, overwrite=True)