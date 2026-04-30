import os


hfd_dir = "/opt/hfd.sh"
local_dir = "./huggingface"

model_names = [
    "BELLE-2/Belle-whisper-large-v3-zh", # https://hf-mirror.com/BELLE-2/Belle-whisper-large-v3-zh
]


if __name__ == "__main__":

    env_cmd = f"export HF_ENDPOINT=https://hf-mirror.com"
    print(f"Running command: {env_cmd}")
    os.system(env_cmd)

    for model_name in model_names:
        download_cmd = f"{hfd_dir} {model_name} --local-dir {local_dir}"
        print(f"Running command: {download_cmd}")
        os.system(download_cmd)