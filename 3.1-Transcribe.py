import os
import argparse

from func.mfa_tools import mfa_align_char, mfa_align_word
from func.configs import N_SYLLABLE, MAX_CONTEXT


# Directories
AUDIO_ROOT = "./data/bids/SMN4Lang/stimuli/audio"
TRANSCRIPTION_ROOT = "./data/bids/SMN4Lang/derivatives/annotations/scripts"
SAVE_ROOT = "./data/saved"


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A script to transcribe audios into text annotaions.")
    parser.add_argument("--char", help="Transcribe audios into char-level annotaions.", action="store_true")
    parser.add_argument("--word", help="Transcribe audios into word-level annotaions.", action="store_true")
    args = parser.parse_args()

    if args.char:
        trans_dir = os.path.join(SAVE_ROOT, "char_transcription")
        os.makedirs(trans_dir, exist_ok=True)

        fnames = [f.split(".")[0] for f in os.listdir(AUDIO_ROOT)]
        for fname in fnames:
            audio_path = os.path.join(AUDIO_ROOT, fname + ".wav")
            trans_path = os.path.join(TRANSCRIPTION_ROOT, fname + ".txt")
            df = mfa_align_char(audio_path, trans_path)
            df = df[df["start"] > MAX_CONTEXT].reset_index(drop=True)
            df.to_csv(os.path.join(trans_dir, fname + ".csv"), index=False)
    
    if args.word:
        trans_dir = os.path.join(SAVE_ROOT, "word_transcription")
        os.makedirs(trans_dir, exist_ok=True)

        fnames = [f.split(".")[0] for f in os.listdir(AUDIO_ROOT)]
        for fname in fnames:
            audio_path = os.path.join(AUDIO_ROOT, fname + ".wav")
            trans_path = os.path.join(TRANSCRIPTION_ROOT, fname + ".txt")
            df = mfa_align_word(audio_path, trans_path)
            df = df[df["start"] > MAX_CONTEXT].reset_index(drop=True)
            df = df[df["word"].str.len() == N_SYLLABLE].reset_index(drop=True)
            df.to_csv(os.path.join(trans_dir, fname + ".csv"), index=False)
    
    if not args.char and not args.word:
        parser.print_help()