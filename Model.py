import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils import convert_indices_to_label

class Model:
    def __init__(self,optimizer='gradientdescent',loss='crossentropy',model_settings=None,model='sanity',train=True,num_hidden=100,num_layers=1):
        self.loss = loss
        self.model_settings = model_settings
        self.model = model
        self.train = train
        self.optimizer = optimizer
        self.num_hidden = num_hidden
        self.num_layers = num_layers

    def get_logits_dropout(self,fingerprint_input,seq_len):
        if self.model == 'sanity':
            return self.sanity_model(fingerprint_input)
        elif self.model == 'baseline':
            return self.baseline_model(fingerprint_input)
        elif self.model == 'vggnet':
            return self.vggnet_model(fingerprint_input)
        elif self.model == 'lstm':
            return self.ctc_lstm_model(fingerprint_input,seq_len)

    def get_in_ground_truth(self):
        seq_len = None
        if self.model=='lstm':
            fingerprint_input = tf.placeholder(tf.float32, [None, None, self.model_settings['dct_coefficient_count']],
                                                    name='fingerprint_input_' + self.model)
            ground_truth_input = tf.sparse_placeholder(tf.int32, name='ground_truth_input_'+ self.model)
            seq_len = tf.placeholder(tf.int32, [None], name='seq_len')
        elif self.model=='vggnet':
            fingerprint_input = tf.placeholder(tf.float32, [None, self.model_settings['fingerprint_size']/self.model_settings['dct_coefficient_count'],self.model_settings['dct_coefficient_count']],
                                                    name='fingerprint_input_' + self.model)
            ground_truth_input = tf.placeholder(tf.int64, [None], name='ground_truth_input')
        else:
            fingerprint_input = tf.placeholder(tf.float32, [None, self.model_settings['fingerprint_size']],
                                                    name='fingerprint_input_' + self.model)
            ground_truth_input = tf.placeholder(tf.int64, [None], name='ground_truth_input')
        return fingerprint_input,ground_truth_input,seq_len

    def get_loss(self, logits, ground_truth, seq_len):
        if self.loss == 'crossentropy':
            return tf.losses.sparse_softmax_cross_entropy(labels=ground_truth, logits=logits)
        elif self.loss == 'ctc':
            return tf.reduce_mean(tf.nn.ctc_loss(ground_truth, logits, seq_len))

    def get_optimizer(self, learning_rate_input, loss_mean):
        if self.optimizer.lower() == 'momentum':
            return tf.train.MomentumOptimizer(learning_rate=learning_rate_input, momentum=0.9).minimize(loss_mean)
        else:
            return tf.contrib.layers.OPTIMIZER_CLS_NAMES[self.optimizer](learning_rate=learning_rate_input).minimize(
                loss_mean)

    def get_confusion_matrix_correct_labels(self,ground_truth_input,logits,seq_len,audio_processor):
        if self.model=='lstm':
            #ctc model
            predicted_indices_orig, _ = tf.nn.ctc_beam_search_decoder(logits, seq_len)
            predicted_indices = convert_indices_to_label(predicted_indices_orig[0], audio_processor)
            # call to utils tensor indices to label self.predicted_indices[0]
            correct_label = convert_indices_to_label(ground_truth_input, audio_processor)
            correct_prediction = tf.equal([predicted_indices], [correct_label])
            confusion_matrix = tf.confusion_matrix([correct_label], [predicted_indices],
                                                        num_classes=self.model_settings['label_count'])
        else:
            predicted_indices = tf.argmax(logits, 1)
            correct_prediction = tf.equal(predicted_indices, ground_truth_input)
            confusion_matrix = tf.confusion_matrix(ground_truth_input, predicted_indices,
                                                        num_classes=self.model_settings['label_count'])
        return predicted_indices,correct_prediction,confusion_matrix

    def sanity_model(self,fingerprint_input):
        if self.train:
            dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
        fingerprint_size = self.model_settings['fingerprint_size']
        label_count = self.model_settings['label_count']
        weights = tf.Variable( tf.truncated_normal([fingerprint_size, label_count], stddev=0.001))
        bias = tf.Variable(tf.zeros([label_count]))
        logits = tf.matmul(fingerprint_input, weights) + bias
        if self.train:
            return logits, dropout_prob
        else:
            return logits

    def baseline_model(self, fingerprint_input):
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

    def vggnet_model(self,fingerprint_input):
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

    def ctc_lstm_model(self,fingerprint_input,seq_len):
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

    @staticmethod
    def conv_2d_relu(prev_node, in_channels, out_channels, filter_width, filter_height, strides=(1, 1, 1, 1), with_bn=False, activation_fn=tf.nn.relu, with_bias=True):
        # filter_shape = [filter_height, filter_width, in_channels, out_channels]
        weights = tf.Variable(
          tf.truncated_normal(
              [filter_height, filter_width, in_channels, out_channels],
              stddev=0.01))
        bias =  tf.Variable(tf.zeros([out_channels]))
        conv = tf.nn.conv2d(prev_node, weights, strides, 'SAME')
        if with_bias:
            conv = tf.nn.bias_add(conv,bias)
        if with_bn:
            bn   = tf.contrib.layers.batch_norm(conv,epsilon=1e-05,renorm_decay=0.1)
        else:
            bn = conv
        activation_out = activation_fn(bn)
        return activation_out




