import math
from utils import find_files, get_parent_folder_name, which_set, stretch, speed,get_file_index
import os
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops
import tensorflow as tf
import numpy as np
import random
import shutil
import librosa


class AudioReader:
    """
    AudioReader class used to read data folder and extract features from wav files available
    mode = test means creating a test set to validate the model
    mode = train means splitting data into training and validation test only
    """
    def __init__(self, audio_dir,model_settings,background_noise,silence_label, unknown_label,classes,augmentation_ops = [],augmentation_percentage = 0,
                 validation_percentage=10,testing_percentage = 10,unknown_percentage = 10,silence_percentage = 10,testing_list = {},validation_list= {},fingerprint_type='mfcc' ,mode='test', train=True):
        print('audio_dir = {}'.format(audio_dir))
        self.model_settings = model_settings
        print('sample_rate = {}'.format(self.model_settings['sample_rate']))
        self.background_data = []
        self.data_index = {}
        self.word_to_index = {}
        self.audio_dir = audio_dir
        self.audio_files = find_files(audio_dir)
        self.background_noise = background_noise
        self.words_list = [silence_label, unknown_label] + classes
        self.augmentation_ops = augmentation_ops
        self.augmentation_percentage = augmentation_percentage
        self.validation_percentage = validation_percentage
        self.testing_percentage = testing_percentage
        self.unknown_percentage = unknown_percentage
        self.fingerprint_type = fingerprint_type
        self.silence_percentage = silence_percentage
        self.testing_list = testing_list
        self.validation_list = validation_list
        print('Found {} files in total in {}.'.format(len(self.audio_files), audio_dir))
        assert len(self.audio_files) != 0
        self.mode = mode
        self.train = train
        self.prepare_data_index()
        if self.train:
            self.prepare_background_data()
        self.prepare_processing_graph()

    def prepare_data_index(self):
        init_dict = {'validation': [], 'testing': [], 'training': []}
        if not (self.train):
            init_dict = {'testing': []}
        elif self.mode == 'train':
            init_dict = {'validation': [], 'training': []}

        self.data_index = init_dict
        unknown_index = init_dict
        all_words = {}
        for wav_file in self.audio_files:
            if self.train:
                word = get_parent_folder_name(wav_file).lower()
            else:
                word = self.words_list[1]  #Unknown label in case of inference
            # used to remove previous augmentation folder
            aug_dir = os.path.join(os.path.dirname(wav_file), 'augmentation')
            if os.path.isdir(aug_dir):
                shutil.rmtree(aug_dir)
            if word == self.background_noise:
                continue
            all_words[word] = True
            if len(self.validation_list)> 0 and len(self.testing_list)>0:
                wav_index = get_file_index(wav_file)
                if wav_index in self.validation_list:
                    set_index = 'validation'
                elif wav_index in self.testing_list:
                    set_index = 'testing'
                else:
                    set_index = 'training'
            else:
                set_index = which_set(wav_file, self.validation_percentage, self.testing_percentage,self.model_settings['max_num_wavs_per_class'])
            if not (self.train):
                set_index = 'testing'  # in case of inference set index will be always testing
            elif self.mode == 'train' and set_index == 'testing':
                # using test set in the training set for production system
                set_index = 'training'
            if word in self.words_list[2:]:
                self.data_index[set_index].append({'label': word, 'file': wav_file})
            else:
                unknown_index[set_index].append({'label': self.words_list[1], 'file': wav_file})
        if self.train:
            for index, wanted_word in enumerate( self.words_list[2:]):
                if wanted_word not in all_words:
                    raise Exception('Expected to find ' + wanted_word +
                                    ' in labels but only found ' +
                                    ', '.join(all_words.keys()))
            silence_wav_path = self.data_index['training'][0]['file']
            for set_index in init_dict:
                set_size = len(self.data_index[set_index])
                silence_size = int(math.ceil(set_size * self.silence_percentage / 100))
                for _ in range(silence_size):
                    self.data_index[set_index].append({
                        'label': self.words_list[0],#silence label
                        'file': silence_wav_path
                    })
                # Pick some unknowns to add to each partition of the data set.
                random.shuffle(
                    unknown_index[set_index])  # TODO might need to get examples from all words and all speakers
                # unknown will be sampled within each mini batch
                unknown_size = int(math.ceil(set_size * self.unknown_percentage / 100))
                self.data_index[set_index].extend(unknown_index[set_index][:unknown_size])

        if self.train:
            self.augment_data('training')
            for set_index in init_dict:
                random.shuffle(self.data_index[set_index])
        for word in all_words:
            if word in self.words_list[2:]:
                self.word_to_index[word] = self.words_list.index(word)
            else:
                self.word_to_index[word] = self.words_list.index(self.words_list[1])
        self.word_to_index[self.words_list[1]] = 1 #unknown label
        self.word_to_index[self.words_list[0]] = 0 #silence label

    def prepare_background_data(self):
        self.background_data = []
        background_dir = os.path.join(self.audio_dir, self.background_noise)
        if not os.path.exists(background_dir):
            return self.background_data
        with tf.Session(graph=tf.Graph()) as sess:
            wav_filename_placeholder = tf.placeholder(tf.string, [])
            wav_loader = io_ops.read_file(wav_filename_placeholder)
            wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
            for wav_path in find_files(background_dir):
                wav_data = sess.run(wav_decoder, feed_dict={wav_filename_placeholder: wav_path}).audio.flatten()
                self.background_data.append(wav_data)
            if not self.background_data:
                raise Exception('No background wav files were found in ' + background_dir)

    def prepare_processing_graph(self):
        """Builds a TensorFlow graph to apply the input distortions.
        Creates a graph that loads a WAVE file, decodes it, scales the volume,
        shifts it in time, adds in background noise, calculates a spectrogram, and
        then builds an MFCC fingerprint from that.
        This must be called with an active TensorFlow session running, and it
        creates multiple placeholder inputs, and one output:
          - wav_filename_placeholder_: Filename of the WAV to load.
          - foreground_volume_placeholder_: How loud the main clip should be.
          - time_shift_padding_placeholder_: Where to pad the clip.
          - time_shift_offset_placeholder_: How much to move the clip in time.
          - background_data_placeholder_: PCM sample data for background noise.
          - background_volume_placeholder_: Loudness of mixed-in background.
          - mfcc_: Output 2D fingerprint of processed audio.
        Args:
          model_settings: Information about the current model being trained.
        """
        desired_samples = self.model_settings['desired_samples']
        self.wav_filename_placeholder_ = tf.placeholder(tf.string, [])

        wav_loader = io_ops.read_file(self.wav_filename_placeholder_)
        wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1, desired_samples=desired_samples)
        # Allow the audio sample's volume to be adjusted.
        self.foreground_volume_placeholder_ = tf.placeholder(tf.float32, [])
        scaled_foreground = tf.multiply(wav_decoder.audio, self.foreground_volume_placeholder_)

        # Shift the sample's start position, and pad any gaps with zeros.
        self.time_shift_padding_placeholder_ = tf.placeholder(tf.int32, [2, 2])
        self.time_shift_offset_placeholder_ = tf.placeholder(tf.int32, [2])
        padded_foreground = tf.pad(scaled_foreground, self.time_shift_padding_placeholder_, mode='CONSTANT')
        sliced_foreground = tf.slice(padded_foreground, self.time_shift_offset_placeholder_, [desired_samples, -1])
        # Mix in background noise.
        self.background_data_placeholder_ = tf.placeholder(tf.float32, [desired_samples, 1])
        self.background_volume_placeholder_ = tf.placeholder(tf.float32, [])
        background_mul = tf.multiply(self.background_data_placeholder_, self.background_volume_placeholder_)
        background_add = tf.add(background_mul, sliced_foreground)
        background_clamp = tf.clip_by_value(background_add, -1.0, 1.0)

        # Run the spectrogram and MFCC ops to get a 2D 'fingerprint' of the audio.
        spectrogram = contrib_audio.audio_spectrogram(background_clamp,
                                                      window_size=self.model_settings['window_size_samples'],
                                                      stride=self.model_settings['window_stride_samples'],
                                                      magnitude_squared=True)
        self.mfcc_ = contrib_audio.mfcc(spectrogram, wav_decoder.sample_rate,
                                        dct_coefficient_count=self.model_settings['dct_coefficient_count'])
        num_spectrogram_bins = spectrogram.shape[-1].value
        lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, self.model_settings['dct_coefficient_count']
        linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
            num_mel_bins, num_spectrogram_bins, self.model_settings['sample_rate'], lower_edge_hertz,
            upper_edge_hertz)
        mel_spectrograms = tf.tensordot(
            spectrogram, linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape(spectrogram.shape[:-1].concatenate(
            linear_to_mel_weight_matrix.shape[-1:]))
        self.mel_ = mel_spectrograms
        self.log_mel_ = tf.log(mel_spectrograms + 1e-6)

    def augment_data(self, set_index):
        """
        :param set_index:  index of the set that needs to be augmented for example training , validation
        :return:  updates self.data_index with augmented data  using the available augmentation ops  and using a specific percentage
        """
        augmented_data = []
        train_data = {}
        for itm in self.data_index[set_index]:
            if 'augmentation' in itm['file']:
                continue
            if itm['label'] in train_data:
                train_data[itm['label']].append(itm['file'])
            else:
                train_data[itm['label']] = [itm['file']]
        n_ops = len(self.augmentation_ops)
        if n_ops==0 or self.augmentation_ops==0:
            return
        for label in train_data:
            random.shuffle(train_data[label])
            train_data[label] = train_data[label][:len(train_data[label]) * self.augmentation_percentage / 100]

            for file in train_data[label]:
                op_index = int(math.floor(random.uniform(0, n_ops)))
                parent_path = os.path.dirname(file)
                new_parent_path = os.path.join(parent_path, 'augmentation')
                aug_wav_path = os.path.join(new_parent_path, os.path.basename(file))

                if not (os.path.isdir(new_parent_path)):
                    os.mkdir(new_parent_path)
                audio = ''
                if self.augmentation_ops[op_index] == 'stretch':
                    if random.random() > 0.5:
                        audio = stretch(file, 0.8,self.model_settings['sample_rate'])
                    else:
                        audio = stretch(file, 1.2,self.model_settings['sample_rate'])
                elif self.augmentation_ops[op_index] == 'speed':
                    speed_rate = np.random.uniform(0.7, 1.3)
                    audio = speed(file, speed_rate,self.model_settings['sample_rate'])

                if not (isinstance(audio, str)):
                    librosa.output.write_wav(aug_wav_path, audio.astype(np.int16), self.model_settings['sample_rate'])
                    augmented_data.append({'label': label, 'file': aug_wav_path})
        self.data_index[set_index].extend(augmented_data)

    def set_size(self, mode):
        """Calculates the number of samples in the dataset partition.
        Args:
          mode: Which partition, must be 'training', 'validation', or 'testing'.
        Returns:
          Number of samples in the partition.
        """
        if mode in self.data_index:
            return len(self.data_index[mode])
        return 0

    def label_to_names(self, preds):
        return [self.words_list[int(p)] for p in np.nditer(preds)]

    def text_to_label(self, text):
        if text not in self.word_to_index:
            return 1
        return self.word_to_index[text]

    def get_data(self, how_many, offset, background_frequency, background_volume_range,
                 time_shift,
                 mode, sess, batch_size, train_step,features_2d = False):
        """

        :param how_many: the amount of data to be returned
        :param offset: offset from the start of the input data
        :param background_frequency: background noise percentage
        :param background_volume_range: volume of the background noise
        :param time_shift: How much to randomly shift the clips by in time.
        :param mode: mode used to retrieve these data
        :param sess: tensorflow session for the preprocessing graph
        :param batch_size: batch size of the requested data
        :param train_step: current training step number
        :param features_2d: returns the wav fingerprint in 2D num_frames * num of DCT components
        :return: the wav data features and the labels
        """
        candidates = self.data_index[mode]
        if how_many == -1:
            sample_count = len(candidates)
        else:
            sample_count = max(0, min(how_many, len(candidates) - offset))

        pick_deterministically = (mode != 'training')
        use_background = self.background_data and (mode == 'training')
        if how_many == -1 or pick_deterministically:
            samples = candidates[offset:offset + sample_count]
        else:
            start_index = (train_step * batch_size) % len(candidates)
            if start_index + sample_count > len(candidates):
                samples = candidates[start_index: ]
                remaining_samples = sample_count-len(samples)
                for i in range(0,remaining_samples):
                    samples.append(candidates[i])

            else:
                samples = candidates[start_index: start_index + sample_count]
            random.shuffle(samples)
        assert (len(candidates) != sample_count)
        f_names = [sample['file'] for sample in samples]
        data, labels = self.featurize_samples(background_frequency, background_volume_range,
                                               offset, samples, sess, time_shift,
                                              use_background,features_2d)
        return data, labels, f_names

    def get_data_random(self, how_many, offset, background_frequency,
                        background_volume_range, time_shift, mode, sess,features_2d=False):
        """
            Args:
          how_many: Desired number of samples to return. -1 means the entire
            contents of this partition.
          offset: Where to start when fetching deterministically.
          model_settings: Information about the current model being trained.
          background_frequency: How many clips will have background noise, 0.0 to
            1.0.
          background_volume_range: How loud the background noise will be.
          time_shift: How much to randomly shift the clips by in time.
          mode: Which partition to use, must be 'training', 'validation', or
            'testing'.
          sess: TensorFlow session that was active when processor was created.
        Returns:
          List of sample data for the transformed samples, and list of label indexes
        """
        candidates = self.data_index[mode]
        if how_many == -1:
            sample_count = len(candidates)
        else:
            sample_count = max(0, min(how_many, len(candidates) - offset))

        f_names = []

        use_background = self.background_data and (mode == 'training')
        pick_deterministically = (mode != 'training')  # to use same validation set
        # Use the processing graph we created earlier to repeatedly to generate the
        # final output sample data we'll use in training.
        samples = []
        for i in range(offset, offset + sample_count):
            if how_many == -1 or pick_deterministically:
                sample_index = i
            else:
                sample_index = np.random.randint(len(candidates))
            sample = candidates[sample_index]
            samples.append(sample)
            f_names.append(sample['file'])
        data, labels = self.featurize_samples(background_frequency, background_volume_range
                                              , offset, samples, sess, time_shift,
                                              use_background,features_2d)
        return data, labels, f_names

    def featurize_samples(self, background_frequency, background_volume_range,
                          offset, samples, sess, time_shift, use_background,features_2d):
        """

        :param background_frequency: percentage of background noise
        :param background_volume_range:  volume of background noise
        :param offset: offset to the start of the data
        :param samples:
        :param sess: TensorFlow session that was active when processor was created.
        :param time_shift: How much to randomly shift the clips by in time.
        :param use_background:  flag to allow blending of background data or not
        :param features_2d:  returns the wav fingerprint in 2D num_frames * num of DCT components
        :return:
        """
        desired_samples = self.model_settings['desired_samples']
        sample_count = len(samples)
        labels = np.zeros(sample_count)
        if features_2d:
            data =  np.zeros((sample_count,self.model_settings['fingerprint_size']/ self.model_settings['dct_coefficient_count'],self.model_settings['dct_coefficient_count']))
        else:
            data = np.zeros((sample_count, self.model_settings['fingerprint_size']))
        for i in range(offset, offset + sample_count):
            sample = samples[i - offset]
            if time_shift > 0:
                time_shift_amount = np.random.randint(-time_shift, time_shift)
            else:
                time_shift_amount = 0
            if time_shift_amount > 0:
                time_shift_padding = [[time_shift_amount, 0], [0, 0]]
                time_shift_offset = [0, 0]
            else:
                time_shift_padding = [[0, -time_shift_amount], [0, 0]]
                time_shift_offset = [-time_shift_amount, 0]
            input_dict = {
                self.wav_filename_placeholder_: sample['file'],
                self.time_shift_padding_placeholder_: time_shift_padding,
                self.time_shift_offset_placeholder_: time_shift_offset
            }
            # Choose a section of background noise to mix in.
            if use_background:
                background_index = np.random.randint(len(self.background_data))
                background_samples = self.background_data[background_index]
                background_offset = np.randint(0, len(background_samples) - self.model_settings['desired_samples'])
                background_clipped = background_samples[background_offset:(background_offset + desired_samples)]
                background_reshaped = background_clipped.reshape([desired_samples, 1])
                if np.random.uniform(0, 1) < background_frequency:
                    background_volume = np.random.uniform(0, background_volume_range)
                else:
                    background_volume = 0
            else:
                background_reshaped = np.zeros([desired_samples, 1])
                background_volume = 0
            input_dict[self.background_data_placeholder_] = background_reshaped
            input_dict[self.background_volume_placeholder_] = background_volume
            if sample['label'] == self.words_list[0]: #Silence label
                input_dict[self.foreground_volume_placeholder_] = 0
            else:
                input_dict[self.foreground_volume_placeholder_] = 1
            # Run the graph to produce the output audio.

            pre_processing_graph = self.mfcc_
            if self.fingerprint_type == 'mel':
                pre_processing_graph = self.mel_
            elif self.fingerprint_type == 'log_mel':
                pre_processing_graph = self.log_mel_
            if features_2d:
                data[i - offset, :, :] =sess.run(pre_processing_graph, feed_dict=input_dict)[0]
            else:
                data[i - offset, :] = sess.run(pre_processing_graph, feed_dict=input_dict).flatten()
            label_index = self.word_to_index[sample['label']]
            labels[i - offset] = label_index
        return data, labels
