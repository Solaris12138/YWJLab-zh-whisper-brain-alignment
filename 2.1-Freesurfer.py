import os
import glob
import tempfile
import re


if __name__ == "__main__":

    current_dir = os.getcwd()
	
    env_cmd = f"export SUBJECTS_DIR={current_dir}/data/bids/SMN4Lang/freesurfer"
    print(f"Running command: {env_cmd}")
    os.system(env_cmd)
    
    t1ws = glob.glob(
        os.path.join(
            f"{current_dir}/data/bids/SMN4Lang",
            "sub-*/anat/sub-*_run-02_T1w.nii.gz"
        )
    )
    
    n_jobs = len(t1ws)
    
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmpf:
        for t1w in t1ws:
            sid = re.search(r"sub-\d+", t1w).group(0)
            tmpf.write(f"{t1w},{sid}\n")
        tmp_path = tmpf.name
    
    parallel_cmd = (
        f"cat {tmp_path} | parallel --colsep ',' --jobs {n_jobs} --eta "
        '"recon-all -i {1} -s {2} -all"'
    )
    
    print(f"Running command: {parallel_cmd}")
    os.system(parallel_cmd)

    os.unlink(tmp_path)