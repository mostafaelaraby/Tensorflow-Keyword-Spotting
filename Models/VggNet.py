import tensorflow as tf
import tensorflow.contrib.slim as slim
from Models.BaseModel import BaseModel


class VggNet(BaseModel):
    def __init__(self, optimizer='gradientdescent', loss='crossentropy', model_settings=None, train=True,
                 num_hidden=100, num_layers=1):
        super(VggNet,self).__init__(optimizer, loss, model_settings, train, num_hidden, num_layers)

    def get_in_ground_truth(self):
        fingerprint_input = tf.placeholder(tf.float32, [None,
                                                        self.model_settings['fingerprint_size'] / self.model_settings[
                                                            'dct_coefficient_count'],
                                                        self.model_settings['dct_coefficient_count']],
                                           name='fingerprint_input_' + self.model)
        ground_truth_input = tf.placeholder(tf.int64, [None], name='ground_truth_input')
        return fingerprint_input, ground_truth_input, None

    def get_logits_dropout(self, fingerprint_input, seq_len):
        """
        Builds a vggnet model based on inout 2d features num_frames * number of bins
        """
        dropout_prob = None
        if self.train:
            dropout_prob = tf.placeholder(tf.float32,name='dropout_prob')
        input_frequency_size=self.model_settings['dct_coefficient_count']
        input_time_size = self.model_settings['spectrogram_length']
        fingerprint_4d = tf.reshape(fingerprint_input, [-1,input_frequency_size, input_time_size, 1])
        conv1a = self.conv_2d_relu(fingerprint_4d,1,8,3,3,strides=(1,1,1, 1),with_bn=True)
        conv1b = self.conv_2d_relu(conv1a,8,8,3,3,strides=(1, 1,1, 1),with_bn=True)
        max_pool_1 = tf.nn.max_pool(conv1b,[1,2,2,1],[1,2,2,1],'SAME')

        if self.train:
            dropout_1 = tf.nn.dropout(max_pool_1,dropout_prob)
        else:
            dropout_1 = max_pool_1
        conv2a = self.conv_2d_relu(dropout_1,8,16,3,3,strides=(1, 1,1, 1),with_bn=True)
        conv2b = self.conv_2d_relu(conv2a,16,16,3,3,strides=(1, 1,1, 1),with_bn=True)
        max_pool_2 = tf.nn.max_pool(conv2b,[1,2,2,1],[1,2,2,1],'SAME')
        if self.train:
            dropout_2 = tf.nn.dropout(max_pool_2,dropout_prob)
        else:
            dropout_2 =  max_pool_2
        conv3a = self.conv_2d_relu(dropout_2,16,32,3,3,strides=(1, 1,1, 1),with_bn=True)
        conv3b = self.conv_2d_relu(conv3a,32,32,3,3,strides=(1, 1,1, 1),with_bn=True)
        max_pool_3 = tf.nn.max_pool(conv3b,[1,2,2,1],[1,2,2,1],'SAME')
        if self.train:
            dropout_3 = tf.nn.dropout(max_pool_3,dropout_prob)
        else:
            dropout_3 = max_pool_3
        prev_conv_shape = dropout_3.get_shape()
        prev_conv_output_width = prev_conv_shape[2]
        prev_conv_output_height = prev_conv_shape[1]
        prev_conv_element_count = int( prev_conv_output_width * prev_conv_output_height * 32)
        flattened_prev_conv = tf.reshape(dropout_3,
                                         [-1, prev_conv_element_count])

        fc_1 = slim.fully_connected(flattened_prev_conv, 512, activation_fn=tf.nn.relu)

        if self.train:
            dropout_4 = tf.nn.dropout(fc_1,dropout_prob)
        else:
            dropout_4 = fc_1
        fc_2 = slim.fully_connected(dropout_4, 256, activation_fn=tf.nn.relu)
        label_count = self.model_settings['label_count']
        logits = slim.fully_connected(fc_2,label_count)
        if self.train:
            return logits,dropout_prob
        else:
            return logits

    def get_confusion_matrix_correct_labels(self, ground_truth_input, logits, seq_len, audio_processor):
        predicted_indices = tf.argmax(logits, 1)
        correct_prediction = tf.equal(predicted_indices, ground_truth_input)
        confusion_matrix = tf.confusion_matrix(ground_truth_input, predicted_indices,
                                                        num_classes=self.model_settings['label_count'])
        return predicted_indices,correct_prediction,confusion_matrix

