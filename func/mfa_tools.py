import os
import re
import subprocess
import tempfile
import pandas as pd
import textgrid
import jieba
import string

from shutil import copy2
from hanziconv import HanziConv


# Punctuations
CN_PUNCS = "，。！？；：“”‘’【】《》（）、—…"


def _clean_text(text: str):
    text = text.replace("\n", "").replace("\r", "")
    return text.strip()


def _split_text_into_sentences(text: str):
    sentences = re.split(r"[。！？!?]", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def _remove_punct(text: str):
    return re.sub(r"[^\u4e00-\u9fff]", "", text)


def mfa_align_word(wav_path, txt_path, dict_path="mandarin_china_mfa", acoustic_model="mandarin_mfa"):
    """
    Using MFA (Montreal Forced Aligner) to transcribe a given audio into word-level annotations.

    Parameters
    ----------
    wav_path : str
        Path to the audio file.
    txt_path : str
        Path to the corresponding text file.
    dict_path : str
        The dictionary name or path used by MFA. The default is a Mandarin dictionary, i.e., "mandarin_china_mfa".
    acoustic_model : str
        The acoustic model name or path used by MFA. The default is the Mandarin acoustic model, i.e., "mandarin_mfa".
    
    Returns
    -------
    df : pandas.DataFrame
        A DataFrame containing three columns: "word", "start", and "end".
    
    """
    wav_path = os.path.abspath(wav_path)
    txt_path = os.path.abspath(txt_path)

    with open(txt_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    raw_text = _clean_text(raw_text)
    raw_text_trad = HanziConv.toTraditional(raw_text)
    words_trad = list(jieba.cut(raw_text_trad))
    punctuation_set = set(string.punctuation + CN_PUNCS)

    filtered_words = []
    for w in words_trad:
        w_stripped = w.strip()
        if not w_stripped: continue        
        is_punct = all(c in punctuation_set for c in w_stripped)
        if not is_punct: filtered_words.append(w_stripped)

    tokenized_text = " ".join(filtered_words)
    if not tokenized_text:
        return pd.DataFrame([], columns=["word", "start", "end"])

    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_dir = os.path.join(tmpdir, "corpus")
        os.makedirs(corpus_dir, exist_ok=True)

        wav_copy = os.path.join(corpus_dir, os.path.basename(wav_path))
        copy2(wav_path, wav_copy)

        lab_copy = os.path.splitext(wav_copy)[0] + ".lab"
        with open(lab_copy, "w", encoding="utf-8") as f:
            f.write(tokenized_text + "\n")

        out_dir = os.path.join(tmpdir, "aligned")
        os.makedirs(out_dir, exist_ok=True)

        cmd = [
            "mfa", "align",
            corpus_dir,
            dict_path,
            acoustic_model,
            out_dir,
            "--clean",
            "--disable_text_normalization",
            "--single_speaker"
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print("Failed to Aligned with error:", e.stderr)
            raise

        tg_path = os.path.join(out_dir, os.path.splitext(os.path.basename(wav_copy))[0] + ".TextGrid")
        if not os.path.exists(tg_path):
            raise FileNotFoundError(f"TextGrid File Not Found: {tg_path}")
        
        tg = textgrid.TextGrid.fromFile(tg_path)
        words_tier = tg.getFirst("words")
        data = []
        
        for interval in words_tier:
            word_trad = interval.mark.strip()
            if not word_trad: continue
                
            if word_trad == "<unk>":
                word_simp = "<unk>" 
            else:
                word_simp = HanziConv.toSimplified(word_trad)
            
            data.append([word_simp, interval.minTime, interval.maxTime])

        df = pd.DataFrame(data, columns=["word", "start", "end"])
        df = df[df["word"] != "<unk>"].reset_index(drop=True)
        return df


def mfa_align_char(wav_path, txt_path, dict_path="mandarin_china_mfa", acoustic_model="mandarin_mfa"):
    """
    Using MFA (Montreal Forced Aligner) to transcribe a given audio into char-level annotations.

    Parameters
    ----------
    wav_path : str
        Path to the audio file.
    txt_path : str
        Path to the corresponding text file.
    dict_path : str
        The dictionary name or path used by MFA. The default is a Mandarin dictionary, i.e., "mandarin_china_mfa".
    acoustic_model : str
        The acoustic model name or path used by MFA. The default is the Mandarin acoustic model, i.e., "mandarin_mfa".
    
    Returns
    -------
    df : pandas.DataFrame
        A DataFrame containing three columns: "char", "start", and "end".
    
    """
    wav_path = os.path.abspath(wav_path)
    txt_path = os.path.abspath(txt_path)

    with open(txt_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    raw_text = _clean_text(raw_text)
    raw_text_trad = HanziConv.toTraditional(raw_text)
    sentences = _split_text_into_sentences(raw_text_trad)
    
    all_chars_trad = []
    for sent in sentences:
        sent_no_punct = _remove_punct(sent)
        if not sent_no_punct:
            continue
        all_chars_trad.extend(list(sent_no_punct))

    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_dir = os.path.join(tmpdir, "corpus")
        os.makedirs(corpus_dir, exist_ok=True)

        wav_copy = os.path.join(corpus_dir, os.path.basename(wav_path))
        copy2(wav_path, wav_copy)

        lab_copy = os.path.splitext(wav_copy)[0] + ".lab"
        with open(lab_copy, "w", encoding="utf-8") as f:
            for sent in sentences:
                sent = _remove_punct(sent)
                if not sent:
                    continue
                f.write(" ".join(list(sent)) + "\n")

        out_dir = os.path.join(tmpdir, "aligned")
        os.makedirs(out_dir, exist_ok=True)

        cmd = [
            "mfa", "align",
            corpus_dir,
            dict_path,
            acoustic_model,
            out_dir,
            "--clean",
            "--disable_text_normalization",
            "--single_speaker"
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print("Failed to Aligned with error:", e.stderr)
            raise

        tg_path = os.path.join(out_dir, os.path.splitext(os.path.basename(wav_copy))[0] + ".TextGrid")
        if not os.path.exists(tg_path):
            raise FileNotFoundError(f"TextGrid File Not Found: {tg_path}")
        
        tg = textgrid.TextGrid.fromFile(tg_path)
        words_tier = tg.getFirst("words")
        data = []
        char_index = 0
        
        for interval in words_tier:
            char_trad = interval.mark.strip()
            if not char_trad:
                continue
                
            if char_trad == "<unk>":
                if char_index < len(all_chars_trad):
                    char_trad = all_chars_trad[char_index]
                else:
                    pass
            
            char_simp = HanziConv.toSimplified(char_trad)
            data.append([char_simp, interval.minTime, interval.maxTime])
            char_index += 1

        df = pd.DataFrame(data, columns=["char", "start", "end"])
        return df


def mfa_align_ipa(wav_path, txt_path, dict_path="mandarin_china_mfa", acoustic_model="mandarin_mfa"):
    """
    Using MFA (Montreal Forced Aligner) to transcribe a given audio into phoneme-level annotations.

    Parameters
    ----------
    wav_path : str
        Path to the audio file.
    txt_path : str
        Path to the corresponding text file.
    dict_path : str
        The dictionary name or path used by MFA. The default is a Mandarin dictionary, i.e., "mandarin_china_mfa".
    acoustic_model : str
        The acoustic model name or path used by MFA. The default is the Mandarin acoustic model, i.e., "mandarin_mfa".
    
    Returns
    -------
    df : pandas.DataFrame
        A DataFrame containing three columns: "phone", "start", and "end".
    
    """
    wav_path = os.path.abspath(wav_path)
    txt_path = os.path.abspath(txt_path)

    with open(txt_path, "r", encoding="utf-8") as f:
        raw_text = _clean_text(f.read())

    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_dir = os.path.join(tmpdir, "corpus")
        os.makedirs(corpus_dir, exist_ok=True)

        wav_copy = os.path.join(corpus_dir, os.path.basename(wav_path))
        copy2(wav_path, wav_copy)

        lab_copy = os.path.splitext(wav_copy)[0] + ".lab"
        with open(lab_copy, "w", encoding="utf-8") as f:
            f.write(raw_text)

        out_dir = os.path.join(tmpdir, "aligned")
        os.makedirs(out_dir, exist_ok=True)

        cmd = [
            "mfa", "align",
            corpus_dir,
            dict_path,
            acoustic_model,
            out_dir,
            "--clean",
            "--disable_text_normalization",
            "--single_speaker"
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print("Failed to Aligned with error:", e.stderr)
            raise

        tg_path = os.path.join(out_dir, os.path.splitext(os.path.basename(wav_copy))[0] + ".TextGrid")
        if not os.path.exists(tg_path):
            raise FileNotFoundError(f"TextGrid File Not Found: {tg_path}")

        tg = textgrid.TextGrid.fromFile(tg_path)
        phones_tier = tg.getFirst("phones")

        phone_data = []
        for interval in phones_tier:
            if interval.mark.strip() == "":
                continue
            phone_data.append({
                "phone": interval.mark.strip(),
                "start": round(interval.minTime, 3),
                "end": round(interval.maxTime, 3)
            })

        return pd.DataFrame(phone_data)