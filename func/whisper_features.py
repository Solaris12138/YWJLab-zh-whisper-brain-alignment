import os
import torch
import torch.nn.functional as F

from transformers import WhisperForConditionalGeneration, WhisperProcessor

from . import whisper
from .utils import check_ffmpeg, SAMPLE_RATE


def load_and_preprocess_audio(audio_path, sampling_rate=SAMPLE_RATE):
    """
    Load a given audio file and perform the whisper preprocessing procedures.

    Parameters
    ----------
    audio_path : str
        Path to the audio file.
    sampling_rate : int
        The sampling rate into which this audio file would be transformed. Default: 16000
    
    Returns
    -------
    audio : np.ndarray
    
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not exists: {audio_path}")
    
    audio = whisper.load_audio(audio_path, sr=sampling_rate)
    return audio


def extract_whisper_features(
    audio_path,
    model,
    processor,
    language="zh",
    crop=None,
    print_transcribe=False
):
    """
    Extract crucial features from whisper.

    Parameters
    ----------
    audio_path : str
        Path to the audio file.
    model : WhisperForConditionalGeneration
        The whisper model.
    processor : WhisperProcessor
        The whisper processor which contains tokenizer and feature extractor.
    language : str
        Clarify the language of the given audio file. Default: "zh"
        See https://github.com/openai/whisper for more supporting languages.
    crop: list
        A list consists of two integers. If not None, the audio would be cropped using the two integers as indices.
    print_transcribe : bool
        If True, the transcription will be printed. Default: False
    
    Returns
    ----------
    features: dict
        A dictionary includes extracted features:
            - mel_spectrogram: np.ndarray, shape (3000, mel_bins)
            - mel_encode_state: np.ndarray, shape (1500, n_features)
            - encoder_last_hidden_state: np.ndarray, shape (1500, n_features)
    
    """
    if not check_ffmpeg():
        raise RuntimeError("Not found ffmpeg. Make sure it is installed correctly")

    if crop:
        if not isinstance(crop, list):
            raise TypeError("The argument 'crop' should be a list consists of two integers.")
        if len(crop) != 2:
            raise ValueError(f"The length of 'crop' should be 2, but got {len(crop)}.")
        if not isinstance(crop[0], int) or not isinstance(crop[-1], int):
            raise TypeError("The argument 'crop' should be a list consists of two integers.")
        if crop[0] >= crop[-1]:
            raise ValueError(f"Invalid crop indices: {crop}")
    
    model.eval()
    device = model.device

    model.config.forced_decoder_ids = (
        processor.tokenizer.get_decoder_prompt_ids(
            language=language, 
            task="transcribe"
        )
    )

    # Load and pre-process the audio file
    audio = load_and_preprocess_audio(audio_path)
    if crop: audio = audio[crop[0]: crop[-1]]
    
    # Extract Log-Mel spectrum
    inputs = processor(
        audio, 
        sampling_rate=SAMPLE_RATE, 
        return_tensors="pt"
    )
    mel = inputs.input_features.to(device) # shape: [1, mel_bins, 3000]
    mel_spectrogram = mel.permute(0, 2, 1).squeeze(0).cpu().numpy()

    # Extract features
    module_outputs = {}
    def _get_module_io(module_name):
        def hook(module, fea_in, fea_out):
            module_outputs[module_name] = fea_out.clone()
        return hook
    
    conv2 = model.model.encoder.conv2
    hook1 = conv2.register_forward_hook(_get_module_io('conv2'))
    
    features = {}
    with torch.no_grad():
        
        outputs = model.generate(
            mel,
            max_length=448,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
        )
        
        mel_encode_state = F.gelu(module_outputs["conv2"])
        mel_encode_state = mel_encode_state.permute(0, 2, 1).squeeze(0).cpu().numpy() # shape: [1500, n_features]
        
        encoder_outputs = outputs.encoder_hidden_states
        encoder_last_hidden_state = encoder_outputs[-1].squeeze(0).cpu().numpy() # shape: [1500, n_features]
        
    if print_transcribe:
        predicted_ids = outputs.sequences
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        print("\n--- 转录结果 (Transcription) ---")
        print("文本:", transcription)
    
    features["mel_spectrogram"] = mel_spectrogram
    features["mel_encode_state"] = mel_encode_state
    features["encoder_last_hidden_state"] = encoder_last_hidden_state
    
    hook1.remove()
    return features


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    language = "zh"
    model_path = "/home/solaris/SemSynProj/In-silico_experiment/huggingface/openai/whisper-tiny"
    audio_path = "/home/solaris/SemSynProj/In-silico_experiment/data/bids/SMN4Lang/stimuli/audio/story_1.wav"

    processor = WhisperProcessor.from_pretrained(model_path, language=language, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    features = extract_whisper_features(
        audio_path=audio_path,
        model=model,
        processor=processor,
        language=language,
        print_transcribe=True
    )