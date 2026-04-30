import mne
import os

from mne.bem import make_watershed_bem, make_scalp_surfaces


if __name__ == "__main__":
    
    spacing = "oct6"
    bem_ico = 5
    
    current_dir = os.getcwd()
    subjects_dir = os.path.join(current_dir, "data/bids/SMN4Lang/freesurfer")
    
    for subject in ["sub-01","sub-02","sub-03","sub-04","sub-05","sub-06",
                    "sub-07","sub-08","sub-09","sub-10","sub-11","sub-12"]:
        
        # Make BEMs using watershed bem
        bem_surf_fname = os.path.join(subjects_dir, subject, "bem",
                                      f"{subject}-ico{bem_ico}-bem.fif")
        bem_sol_fname = os.path.join(subjects_dir, subject, "bem",
                                     f"{subject}-ico{bem_ico}-bem-sol.fif")
        src_fname = os.path.join(subjects_dir, subject, "bem",
                                 f"{subject}-ico{bem_ico}-src.fif")
        
        make_watershed_bem(subject,
                           subjects_dir=subjects_dir,
                           overwrite=True,
                           show=False,
                           verbose=False)
        
        make_scalp_surfaces(subject=subject,
                            subjects_dir=subjects_dir,
                            force=True,
                            overwrite=True)

        # make BEM models
        bem_surf = mne.make_bem_model(
            subject,
            ico=bem_ico,
            conductivity=[0.3],
            subjects_dir=subjects_dir
        )
        mne.write_bem_surfaces(bem_surf_fname, bem_surf)

        # make BEM solution
        bem_sol = mne.make_bem_solution(bem_surf)
        mne.write_bem_solution(bem_sol_fname, bem_sol)

        # Create the surface source space
        src = mne.setup_source_space(subject, spacing, subjects_dir=subjects_dir)
        mne.write_source_spaces(src_fname, src, overwrite=True)