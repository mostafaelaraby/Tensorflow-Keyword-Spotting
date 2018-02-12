import os
import re
import hashlib
from tensorflow.python.util import compat
import librosa
import numpy as np
import cv2
import shutil


speaker_pattern = re.compile(r'([^_/]+)_\w+\.wav$')

def truncate_dir(in_dir):
    if os.path.isdir(in_dir):
        shutil.rmtree(in_dir)
    os.mkdir(in_dir)


def get_recursive_files(folderPath, regex):
    results = os.listdir(folderPath)
    out_files = []
    cnt_files = 0
    for file in results:
        if os.path.isdir(os.path.join(folderPath, file)):
            out_files += get_recursive_files(os.path.join(folderPath, file), regex)
        elif re.match(regex, file,  re.I):  # file.startswith(startExtension) or file.endswith(".txt") or file.endswith(endExtension):
            out_files.append(os.path.join(folderPath, file))
            cnt_files = cnt_files + 1
    return out_files


def get_file_index(file_path):
    new_filename = file_path
    parent = get_parent_folder_name(new_filename)
    if parent == 'augmentation':
        new_filename = file_path.replace('augmentation', '').replace('//', '/')
    parent = get_parent_folder_name(new_filename)
    file_index = parent + '/' + os.path.basename(new_filename)
    return file_index


def find_files(directory, pattern='.*\.wav'):
    """Recursively finds all files matching the pattern."""
    files = sorted(
        get_recursive_files(directory, pattern))  # glob.iglob(os.path.join(directory, pattern), recursive=True))
    return [file for file in files if
            get_parent_folder_name(file) != 'augmentation']  # todo remove augmentation check condition


def load_audio_file(file_path,sample_rate):
    input_length = sample_rate
    data = librosa.core.load(file_path, sr=sample_rate)[0]  # , sr=16000
    if len(data) > input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
    return data


def stretch(wav_file, factor,sample_rate):
    wav = load_audio_file(wav_file,sample_rate)
    input_length = sample_rate
    data = librosa.effects.time_stretch(wav, factor)
    if len(data) > input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
    return data


def speed(wav_file, speed_rate,sample_rate):
    wav = load_audio_file(wav_file,sample_rate)
    input_length = sample_rate
    wav_speed_tune = cv2.resize(wav, (1, int(len(wav) * speed_rate))).squeeze()
    if len(wav_speed_tune) > input_length:
        wav_speed_tune = wav_speed_tune[:input_length]
    else:
        wav_speed_tune = np.pad(wav_speed_tune, (0, max(0, input_length - len(wav_speed_tune))), "constant")
    return wav_speed_tune


def extract_speaker_id(filename):
    global speaker_pattern
    match_goups = speaker_pattern.search(filename)
    return match_goups.group(1)


def get_parent_folder_name(filename):
    return os.path.split(os.path.dirname(filename))[1].strip().lower()


def which_set(filename, validation_percentage, testing_percentage,max_num_wavs_per_class):
    hash_name = extract_speaker_id(filename)  # to split based on speaker
    hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) %
                        (max_num_wavs_per_class + 1)) *
                       (100.0 / max_num_wavs_per_class))
    if percentage_hash < validation_percentage:
        return 'validation'
    elif percentage_hash < (testing_percentage + validation_percentage):
        return 'testing'
    else:
        return 'training'

def model_factory_to_object(model_name):
    import glob
    avail_models = [model.replace('.py','').replace('Models/','') for model in glob.glob('Models/*.py') if 'init' not in model and 'basemodel' not in model]
    class_name = 'SanityModel'
    for model in avail_models:
        if model_name in model.lower():
            class_name = model
    import importlib
    model_ = getattr(importlib.import_module('Models.'+class_name), class_name)
    return model_


def prepare_model_settings(label_count, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms,
                           dct_coefficient_count,time_shift_ms,max_num_wavs_per_class):
    """Calculates common settings needed for all models.
      Args:
        label_count: How many classes are to be recognized.
        sample_rate: Number of audio samples per second.
        clip_duration_ms: Length of each audio clip to be analyzed.
        window_size_ms: Duration of frequency analysis window.
        window_stride_ms: How far to move in time between frequency windows.
        dct_coefficient_count: Number of frequency bins to use for analysis.
      Returns:
        Dictionary containing common settings.
      """
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)
    length_minus_window = (desired_samples - window_size_samples)
    time_shift_samples = int((time_shift_ms * sample_rate) / 1000)
    if length_minus_window < 0:
        spectrogram_length = 0
    else:
        spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
    fingerprint_size = dct_coefficient_count * spectrogram_length
    return {
        'desired_samples': desired_samples,
        'window_size_samples': window_size_samples,
        'window_stride_samples': window_stride_samples,
        'spectrogram_length': spectrogram_length,
        'dct_coefficient_count': dct_coefficient_count,
        'fingerprint_size': fingerprint_size,
        'label_count': label_count,
        'sample_rate': sample_rate,
        'window_size_ms': window_size_ms,
        'window_stride_ms': window_stride_ms,
        'time_shift_samples': time_shift_samples,
        'max_num_wavs_per_class': max_num_wavs_per_class
    }

