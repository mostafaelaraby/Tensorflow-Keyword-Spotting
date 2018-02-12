from Models.BaseModel import BaseModel
import tensorflow as tf
import numpy as np
from utils import get_file_index


class CTCModelLSTM(BaseModel):
    def __init__(self, optimizer='gradientdescent', loss='crossentropy', model_settings=None, train=True,
                 num_hidden=100, num_layers=1):
        super(CTCModelLSTM,self).__init__(optimizer, loss, model_settings, train, num_hidden, num_layers)
        self.SPACE_TOKEN = '<space>'
        self.SPACE_INDEX = 0
        self.FIRST_INDEX = ord('a') - 1
        self.reader = None

    def get_in_ground_truth(self):
        fingerprint_input = tf.placeholder(tf.float32, [None, None, self.model_settings['dct_coefficient_count']],
                                           name='fingerprint_input_' + self.model)
        ground_truth_input = tf.sparse_placeholder(tf.int32, name='ground_truth_input_' + self.model)
        seq_len = tf.placeholder(tf.int32, [None], name='seq_len')
        return fingerprint_input, ground_truth_input, seq_len

    def get_logits_dropout(self, fingerprint_input, seq_len):
        num_classes = ord('z') - ord('a') + 1 + 1 + 1
        cells = [tf.contrib.rnn.LSTMCell(self.num_hidden, state_is_tuple=True) for i in range(self.num_layers)]
        stack = tf.contrib.rnn.MultiRNNCell(cells,
                                            state_is_tuple=True)
        outputs, _ = tf.nn.dynamic_rnn(stack, fingerprint_input, seq_len, dtype=tf.float32)
        shape = tf.shape(fingerprint_input)
        batch_s, max_time_steps = shape[0], shape[1]
        outputs = tf.reshape(outputs, [-1, self.num_hidden])
        W = tf.Variable(tf.truncated_normal([self.num_hidden,
                                             num_classes],
                                            stddev=0.1))
        b = tf.Variable(tf.constant(0., shape=[num_classes]))
        logits = tf.matmul(outputs, W) + b
        logits = tf.reshape(logits, [batch_s, -1, num_classes])
        logits = tf.transpose(logits, (1, 0, 2))
        if self.train:
            return logits,None
        else:
            return logits

    def get_confusion_matrix_correct_labels(self, ground_truth_input, logits, seq_len, audio_processor):
        predicted_indices_orig, _ = tf.nn.ctc_beam_search_decoder(logits, seq_len)
        predicted_indices = self.convert_indices_to_label(predicted_indices_orig[0], audio_processor)
        # call to utils tensor indices to label self.predicted_indices[0]
        correct_label = self.convert_indices_to_label(ground_truth_input, audio_processor)
        correct_prediction = tf.equal([predicted_indices], [correct_label])
        confusion_matrix = tf.confusion_matrix([correct_label], [predicted_indices],
                                               num_classes=self.model_settings['label_count'])
        return predicted_indices,correct_prediction,confusion_matrix

    def convert_indices_to_label(self,predicted_tensor, audio_reader):
        self.reader = audio_reader
        predicted_text =  tf.py_func(self.char_indices_to_label, [predicted_tensor.values], tf.string)
        return tf.py_func(audio_reader.text_to_label,[predicted_text],tf.int64)

    def char_indices_to_label(self,value):
        predicted_text = ''.join([chr(x) for x in np.asarray(value) + self.FIRST_INDEX])
        predicted_text = predicted_text.replace(chr(ord('z') + 1), '').replace(chr(ord('a') - 1), ' ').strip()

        if predicted_text == '':
            predicted_text = self.reader.words_list[0]
        elif predicted_text in self.reader.words_list[2:]:
            pass
        else:
            predicted_text = self.reader.words_list[1]
        return predicted_text

    def convert_batch_to_ctc_format(self,inputs, fnames, audio_reader):
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
            target = target.replace(' ', '  ')  # to add space token in case of any ome found
            target = target.split(' ')
            # Adding blank label
            target = np.hstack([self.SPACE_TOKEN if x == '' else list(x) for x in target])
            # Transform char into index
            target = np.asarray([self.SPACE_INDEX if x == self.SPACE_TOKEN else ord(x) - self.FIRST_INDEX for x in target])

            # Creating sparse representation to feed the placeholder
            target = self.sparse_tuple_from([target])
            train_targets.append(target)
            train_inputs.append(train_input)
            train_inputs_len.append(train_seq_len)
        return train_inputs, train_targets, train_inputs_len

    def sparse_tuple_from(self,sequences, dtype=np.int32):
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

