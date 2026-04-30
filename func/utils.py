import os
import mne
import torch
import psutil
import random
import subprocess
import numpy as np

from operator import add
from functools import reduce
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# hard-coded audio hyperparameters
SAMPLE_RATE = 16000 # frequency (Hz)
CHUNK_LENGTH = 30 # seconds
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk


def check_ffmpeg():
    """
    Check whether ffmpeg is installed correctly.
    """
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_memory():
    """
    Check available system memory
    """
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"当前进程内存使用: {memory_info.rss / 1024 / 1024:.2f} MB")
    
    system_memory = psutil.virtual_memory()
    print(f"系统可用内存: {system_memory.available / 1024 / 1024:.2f} MB")


def set_seed(seed):
    """
    Set random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def combine_labels(subject, subjects_dir, parc, label_names):
    """
    Combine the labels of selected regions.

    Parameters
    ----------
    subject : str
        The name of subject of whom the source space is.
    subjects_dir : str | path-like
        The path to the directory containing the FreeSurfer subjects reconstructions.
    parc : str
        The parcellation to use, e.g., 'aparc' or 'aparc.a2009s'.
    label_names : list of str
        A list consists of the names of selected regions.
    
    Returns
    -------
    label : mne.Label

    """

    labels =  mne.read_labels_from_annot(
        subject=subject, 
        parc=parc, 
        subjects_dir=subjects_dir, 
        hemi="both", 
        regexp=None
    )
    labels = [label if label.name in label_names else None for label in labels] 
    labels = list(filter(None, labels))
    label = reduce(add, labels)

    return label


def PCA_reduction(data, n_components, if_scale=True):
    """
    Using PCA for dimensionality reduction

    Parameters
    ----------
    data : numpy.ndarray
        The original data. Shape: (n_samples, n_timepoints, n_features)
    n_components : int | float
        Number of components to keep.
    if_scale : bool
        If True, the data would be standardized.
    
    Returns
    -------
    data : numpy.ndarray
        The reduced data. Shape: (n_samples, n_timepoints, n_reduced_features)

    """
    N, T, D = data.shape

    data = data.reshape(-1, D)

    if if_scale:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

    pca = PCA(n_components=n_components)
    data = pca.fit_transform(data)

    data = data.reshape(N, T, -1)
    return data