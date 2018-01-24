import os
import re
import hashlib
from tensorflow.python.util import compat
import librosa
import numpy as np
import cv2
import tensorflow as tf
import shutil

SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1
speaker_pattern = re.compile(r'([^_/]+)_\w+\.wav$')
reader = None

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


def convert_batch_to_ctc_format(inputs, fnames,audio_reader):
    """
    Converts batch to ctc format
    :param input sequences
    :param filenames used to get parent folder holding label text
    :return: ctc formatted data for ctc training
    """
    train_inputs = []
    train_targets = []
    train_inputs_len = []
    for input_index, input_fingerprint in enumerate(inputs):
        # Transform in 3D array
        train_input = input_fingerprint
        train_input = np.asarray(train_input[np.newaxis, :])
        train_input = (train_input - np.mean(train_input)) / np.std(train_input)
        train_seq_len = [train_input.shape[1]]
        target = get_file_index(fnames[input_index]).split('/')[0].strip()
        target = target.replace(' ','  ')# to add space token in case of any ome found
        target = target.split(' ')
        # Adding blank label
        target = np.hstack([SPACE_TOKEN if x == '' else list(x) for x in target])
        # Transform char into index
        target = np.asarray([SPACE_INDEX if x == SPACE_TOKEN else ord(x) - FIRST_INDEX for x in target])

        # Creating sparse representation to feed the placeholder
        target = sparse_tuple_from([target])
        train_targets.append(target)
        train_inputs.append(train_input)
        train_inputs_len.append(train_seq_len)
    return train_inputs, train_targets, train_inputs_len


def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
        dtype: type of output sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


def convert_indices_to_label(predicted_tensor, audio_reader):
    global reader
    reader = audio_reader
    predicted_text = tf.py_func(char_indices_to_label, [predicted_tensor.values], tf.string)
    return tf.py_func(reader.text_to_label, [predicted_text], tf.int64)


def char_indices_to_label(value):
    global reader
    predicted_text = ''.join([chr(x) for x in np.asarray(value) + FIRST_INDEX])
    predicted_text = predicted_text.replace(chr(ord('z') + 1), '').replace(chr(ord('a') - 1), ' ').strip()

    if predicted_text == '':
        predicted_text = reader.words_list[0]
    elif predicted_text in reader.words_list[2:]:
        pass
    else:
        predicted_text = reader.words_list[1]
    return predicted_text
