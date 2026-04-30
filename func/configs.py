MAX_CONTEXT = 10 # sec
MAX_TIMELAG = 0.5 # sec

WHISPER_SEG_LEN = 20 # ms
CONTEXT_LEN = [
    0, 20, 40, 60, 80,
    100, 200, 400, 600, 800,
    1000, 2000, 4000, 6000, 8000,
    int(MAX_CONTEXT * 1000)
] # ms

RESAMPLE_SFREQ = 1000 # Hz
DECIM = 5

DURATION_CHAR = 0.2 # sec
DURATION_WORD = 0.4 # sec

N_SYLLABLE = 2

reject_criteria = dict(
    grad=4000e-13,  # unit: T / m (gradiometers)
    mag=4000e-15,   # unit: T (magnetometers)
)

model_paths = {
    "tiny" : "huggingface/openai/whisper-tiny",
    "base" : "huggingface/openai/whisper-base",
    "small" : "huggingface/openai/whisper-small",
    "medium" : "huggingface/openai/whisper-medium",
    "large" : "huggingface/openai/whisper-large-v3",
    "zh-ft" : "huggingface/BELLE-2/Belle-whisper-large-v3-zh"
}

ATLAS = "aparc.a2009s"

AUDLANG_NET = [
    "Lat_Fis-post-lh",
    "G_temp_sup-Plan_tempo-lh",
    "S_temporal_transverse-lh",
    "S_temporal_sup-lh",
    "G_temp_sup-Lateral-lh",
    "S_interm_prim-Jensen-lh",
    "G_temporal_middle-lh",
    "S_temporal_inf-lh",
    "G_pariet_inf-Angular-lh",
    "G_pariet_inf-Supramar-lh",
    "Pole_temporal-lh",
    "G_and_S_subcentral-lh",
    "G_front_inf-Triangul-lh",
    "S_front_inf-lh",
    "S_orbital_lateral-lh",
    "G_front_inf-0rbital-lh",
    "S_precentral-inf-part-lh",
    "G_front_inf-0percular-lh",
    "Lat_Fis-ant-Vertical-lh"
]