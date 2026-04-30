import os
import argparse
import openneuro as on


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="The script to download data for pre-training encoding model (MEG version).")

    parser.add_argument("--n_jobs", default=5, help="The maximum number of downloads to run in parallel.")
    
    args = parser.parse_args()
    
    # For SMN4Lang
    bids_root = f"./data/bids/SMN4Lang"
    if not os.path.exists(bids_root):
        os.makedirs(bids_root, exist_ok=True)
    
    on.download(
        dataset="ds004078",
        target_dir=bids_root,
        include=[
            "README",
            "participants.json",
            "participants.tsv",
            "dataset_description.json",
            "sub-*/anat/**",
            "derivatives/annotations",
            "derivatives/preprocessed_data/sub-*/MEG/**",
            "derivatives/annotations/embeddings",
            "stimuli/audio"
        ],
        max_concurrent_downloads=args.n_jobs
    )