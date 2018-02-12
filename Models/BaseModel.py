import tensorflow as tf
import abc


class BaseModel(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self,optimizer='gradientdescent',loss='crossentropy',model_settings=None,train=True,num_hidden=100,num_layers=1):
        self.loss = loss
        self.model_settings = model_settings
        self.train = train
        self.optimizer = optimizer
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.model = 'model_name'

    @abc.abstractmethod
    def get_in_ground_truth(self):
        pass

    @abc.abstractmethod
    def get_logits_dropout(self, fingerprint_input, seq_len):
        pass

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

    @abc.abstractmethod
    def get_confusion_matrix_correct_labels(self, ground_truth_input, logits, seq_len, audio_processor):
        pass

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