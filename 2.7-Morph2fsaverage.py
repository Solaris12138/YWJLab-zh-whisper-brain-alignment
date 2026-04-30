import os
import glob
import mne


if __name__ == "__main__":

    # We set spacing as ico3, ico4 and ico5.
    # Cause Nilearn only provides ico# meshes for fsaverage.
    # See "nilearn.datasets.load_fsaverage" for more details.
    # https://nilearn.github.io/dev/modules/generated/nilearn.datasets.load_fsaverage.html#nilearn.datasets.load_fsaverage
    for ico in [3, 4, 5]:
        spacing = f"ico{ico}"

        derivatives_root = "./data/bids/SMN4Lang/derivatives/preprocessed_data"
        subjects_dir = "./data/bids/SMN4Lang/freesurfer"
        subject_to = "fsaverage"

        src_to = mne.read_source_spaces(
            os.path.join(subjects_dir,
                         f"{subject_to}/bem/{subject_to}-{spacing}-src.fif")
        )

        for subject in os.listdir(subjects_dir):
            if not subject.startswith("sub-"):
                continue

            fwd_root = os.path.join(derivatives_root, f"{subject}/MEG/fwd")
            fwd_fname = glob.glob(os.path.join(
                derivatives_root,
                f"{subject}/MEG/fwd/{subject}_task-RDR_run-*-fwd.fif"
            ))[0]
            fwd = mne.read_forward_solution(fwd_fname)
            
            src = fwd["src"]

            morph = mne.compute_source_morph(
                src=src,
                src_to=src_to,
                subject_from=subject,
                subject_to=subject_to,
                subjects_dir=subjects_dir,
                spacing=ico
            )

            morph_fname = os.path.join(subjects_dir, subject, "bem",
                                       f"{subject}2{subject_to}-{spacing}")
            morph.save(morph_fname, overwrite=True)