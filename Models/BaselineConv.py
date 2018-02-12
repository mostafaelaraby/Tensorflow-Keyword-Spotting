import tensorflow as tf
from Models.BaseModel import BaseModel


class BaselineConv(BaseModel):
    def __init__(self, optimizer='gradientdescent', loss='crossentropy', model_settings=None, train=True,
                 num_hidden=100, num_layers=1):
        super(BaselineConv,self).__init__(optimizer, loss, model_settings, train, num_hidden, num_layers)


    def get_in_ground_truth(self):
        fingerprint_input = tf.placeholder(tf.float32, [None, self.model_settings['fingerprint_size']],
                                           name='fingerprint_input_' + self.model)
        ground_truth_input = tf.placeholder(tf.int64, [None], name='ground_truth_input')
        return fingerprint_input, ground_truth_input, None

    def get_logits_dropout(self, fingerprint_input, seq_len):
        """Builds a standard convolutional model.
          This is roughly the network labeled as 'cnn-trad-fpool3' in the
          'Convolutional Neural Networks for Small-footprint Keyword Spotting' paper:
          http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf
        """
        if self.train:
            dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
        input_frequency_size = self.model_settings['dct_coefficient_count']
        input_time_size = self.model_settings['spectrogram_length']
        fingerprint_4d = tf.reshape(fingerprint_input, [-1, input_frequency_size,input_time_size, 1])
        first_filter_width = 8
        first_filter_height = 20
        first_filter_count = 64
        first_relu = self.conv_2d_relu(fingerprint_4d, 1, first_filter_count, first_filter_width, first_filter_height)
        if self.train:
            first_dropout = tf.nn.dropout(first_relu, dropout_prob)
        else:
            first_dropout = first_relu
        max_pool = tf.nn.max_pool(first_dropout, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        second_filter_width = 4
        second_filter_height = 10
        second_filter_count = 64
        second_relu = self.conv_2d_relu(max_pool, first_filter_count, second_filter_count, second_filter_width,
                                        second_filter_height)
        if self.train:
            second_dropout = tf.nn.dropout(second_relu, dropout_prob)
        else:
            second_dropout = second_relu
        second_conv_shape = second_dropout.get_shape()
        second_conv_output_width = second_conv_shape[2]
        second_conv_output_height = second_conv_shape[1]
        second_conv_element_count = int(
            second_conv_output_width * second_conv_output_height *
            second_filter_count)
        flattened_second_conv = tf.reshape(second_dropout,
                                           [-1, second_conv_element_count])
        label_count = self.model_settings['label_count']
        final_fc_weights = tf.Variable(
            tf.truncated_normal(
                [second_conv_element_count, label_count], stddev=0.01))
        final_fc_bias = tf.Variable(tf.zeros([label_count]))
        final_fc = tf.matmul(flattened_second_conv, final_fc_weights)
        final_fc = tf.nn.bias_add(final_fc, final_fc_bias)
        if self.train:
            return final_fc, dropout_prob
        else:
            return final_fc

    def get_confusion_matrix_correct_labels(self, ground_truth_input, logits, seq_len, audio_processor):
        predicted_indices = tf.argmax(logits, 1)
        correct_prediction = tf.equal(predicted_indices, ground_truth_input)
        confusion_matrix = tf.confusion_matrix(ground_truth_input, predicted_indices,
                                                        num_classes=self.model_settings['label_count'])
        return predicted_indices,correct_prediction,confusion_matrix

