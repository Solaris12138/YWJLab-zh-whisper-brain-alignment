import os
import mne
import warnings
warnings.filterwarnings("error")


if __name__ == "__main__":

    root = "./data/bids/SMN4Lang/derivatives/preprocessed_data"

    failed_files = list()

    for sub in os.listdir(root):
        sub_dir = os.path.join(root, f"{sub}/MEG")
        for file in os.listdir(sub_dir):
            if ".fif" in file:
                try:
                    raw = mne.io.read_raw_fif(os.path.join(sub_dir, file))
                except:
                    failed_files.append(os.path.join(sub_dir, file))

    with open("./check_results.tsv", "w") as f:
        for file in failed_files:
            f.write(file)
            f.write("\n")