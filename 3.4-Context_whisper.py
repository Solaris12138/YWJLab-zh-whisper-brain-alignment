import os
import torch
import logging
import argparse
import numpy as np
import pandas as pd

from transformers import WhisperProcessor, WhisperForConditionalGeneration

from func.utils import SAMPLE_RATE
from func.configs import model_paths, DURATION_CHAR, DURATION_WORD, WHISPER_SEG_LEN, CONTEXT_LEN
from func.whisper_features import extract_whisper_features


# Directories
BIDS_ROOT = "./data/bids/SMN4Lang"
AUDIO_ROOT = os.path.join(BIDS_ROOT, "stimuli/audio")
SAVE_ROOT = "./data/saved"

# Global Parameters
LANG = "zh"


if __name__ == "__main__":
    """
    这段代码可以并行优化一下, 但我懒得写了...
    This piece of code can be optimized in parallel, but I'm too lazy to write it...
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )

    current_dir = os.getcwd()

    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    parser = argparse.ArgumentParser(description="A script to extract Whisper features for chars and words.")
    parser.add_argument("--char", help="Extract Whisper features for char-level.", action="store_true")
    parser.add_argument("--word", help="Extract Whisper features for word-level.", action="store_true")
    args = parser.parse_args()

    if args.char:
        logging.info("--- Extracting Whisper Features for Char-Level ---")
        DURATION = DURATION_CHAR
        trans_dir = os.path.join(SAVE_ROOT, "char_transcription")
        save_root = os.path.join(SAVE_ROOT, "whisper_features/char-level")

        for model_name, model_path in model_paths.items():
            logging.info(f"Using Model: whisper-{model_name}")
            model_path = os.path.join(current_dir, model_path)
            save_dir = os.path.join(save_root, f"{model_name}")
            os.makedirs(save_dir, exist_ok=True)

            processor = WhisperProcessor.from_pretrained(model_path, language=LANG, task="transcribe")
            model = WhisperForConditionalGeneration.from_pretrained(model_path)
            model.to(device)
            model.eval()

            for file in os.listdir(trans_dir):
                logging.info(f"Processing: {file}")
                run = file.split("_")[-1].replace(".csv", "")

                df = pd.read_csv(os.path.join(trans_dir, file))
                start_times = df["start"].tolist()

                audio_fname = os.path.join(AUDIO_ROOT, file.replace(".csv", ".wav"))

                for context_len in CONTEXT_LEN:
                    context_len_sec = context_len / 1000
                    
                    mel_spec = list()
                    conv_emb = list()
                    enc_emb = list()
                    for start_time in start_times:
                        modified_start_idx = int((start_time - context_len_sec) * SAMPLE_RATE)
                        end_idx = int((start_time + DURATION) * SAMPLE_RATE)

                        features = extract_whisper_features(
                            audio_path=audio_fname,
                            model=model,
                            processor=processor,
                            language=LANG,
                            crop=[modified_start_idx, end_idx],
                            print_transcribe=False
                        )
                        
                        mel_spec.append(features["mel_spectrogram"])
                        conv_emb.append(features["mel_encode_state"])
                        enc_emb.append(features["encoder_last_hidden_state"])
                    
                    mel_spec = np.asarray(mel_spec, dtype=np.float32)
                    conv_emb = np.asarray(conv_emb, dtype=np.float32)
                    enc_emb = np.asarray(enc_emb, dtype=np.float32)

                    crop_start = context_len // WHISPER_SEG_LEN
                    crop_end = crop_start + int(DURATION * 1000 / WHISPER_SEG_LEN)
                    
                    np.save(os.path.join(save_dir, f"whisper_mel_context-{context_len}_{file.split('.')[0]}.npy"), 
                            mel_spec[:, crop_start:crop_end, :])
                    np.save(os.path.join(save_dir, f"whisper_acoustics_context-{context_len}_{file.split('.')[0]}.npy"), 
                            conv_emb[:, crop_start:crop_end, :])
                    np.save(os.path.join(save_dir, f"whisper_speech_context-{context_len}_{file.split('.')[0]}.npy"), 
                            enc_emb[:, crop_start:crop_end, :])

    if args.word:
        logging.info("--- Extracting Whisper Features for Word-Level ---")
        DURATION = DURATION_WORD
        trans_dir = os.path.join(SAVE_ROOT, "word_transcription")
        save_root = os.path.join(SAVE_ROOT, "whisper_features/word-level")

        for model_name, model_path in model_paths.items():
            logging.info(f"Using Model: whisper-{model_name}")
            model_path = os.path.join(current_dir, model_path)
            save_dir = os.path.join(save_root, f"{model_name}")
            os.makedirs(save_dir, exist_ok=True)

            processor = WhisperProcessor.from_pretrained(model_path, language=LANG, task="transcribe")
            model = WhisperForConditionalGeneration.from_pretrained(model_path)
            model.to(device)
            model.eval()

            for file in os.listdir(trans_dir):
                logging.info(f"Processing: {file}")

                df = pd.read_csv(os.path.join(trans_dir, file))
                start_times = df["start"].tolist()

                audio_fname = os.path.join(AUDIO_ROOT, file.replace(".csv", ".wav"))

                for context_len in CONTEXT_LEN:
                    context_len_sec = context_len / 1000
                    
                    mel_spec = list()
                    conv_emb = list()
                    enc_emb = list()
                    for start_time in start_times:
                        modified_start_idx = int((start_time - context_len_sec) * SAMPLE_RATE)
                        end_idx = int((start_time + DURATION) * SAMPLE_RATE)

                        features = extract_whisper_features(
                            audio_path=audio_fname,
                            model=model,
                            processor=processor,
                            language=LANG,
                            crop=[modified_start_idx, end_idx],
                            print_transcribe=False
                        )
                        
                        mel_spec.append(features["mel_spectrogram"])
                        conv_emb.append(features["mel_encode_state"])
                        enc_emb.append(features["encoder_last_hidden_state"])
                    
                    mel_spec = np.asarray(mel_spec, dtype=np.float32)
                    conv_emb = np.asarray(conv_emb, dtype=np.float32)
                    enc_emb = np.asarray(enc_emb, dtype=np.float32)

                    crop_start = context_len // WHISPER_SEG_LEN
                    crop_end = crop_start + int(DURATION * 1000 / WHISPER_SEG_LEN)
                    
                    np.save(os.path.join(save_dir, f"whisper_mel_context-{context_len}_{file.split('.')[0]}.npy"), 
                            mel_spec[:, crop_start:crop_end, :])
                    np.save(os.path.join(save_dir, f"whisper_acoustics_context-{context_len}_{file.split('.')[0]}.npy"), 
                            conv_emb[:, crop_start:crop_end, :])
                    np.save(os.path.join(save_dir, f"whisper_speech_context-{context_len}_{file.split('.')[0]}.npy"), 
                            enc_emb[:, crop_start:crop_end, :])
    
    if not args.char and not args.word:
        parser.print_help()