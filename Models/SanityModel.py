import tensorflow as tf
from Models.BaseModel import BaseModel


class SanityModel(BaseModel):
    def __init__(self, optimizer='gradientdescent', loss='crossentropy', model_settings=None, train=True,
                 num_hidden=100, num_layers=1):
        super(SanityModel,self).__init__(optimizer, loss, model_settings, train, num_hidden, num_layers)

    def get_in_ground_truth(self):
        fingerprint_input = tf.placeholder(tf.float32, [None, self.model_settings['fingerprint_size']],
                                           name='fingerprint_input_' + self.model)
        ground_truth_input = tf.placeholder(tf.int64, [None], name='ground_truth_input')
        return fingerprint_input, ground_truth_input, None

    def get_logits_dropout(self, fingerprint_input, seq_len):
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

    def get_confusion_matrix_correct_labels(self, ground_truth_input, logits, seq_len, audio_processor):
        predicted_indices = tf.argmax(logits, 1)
        correct_prediction = tf.equal(predicted_indices, ground_truth_input)
        confusion_matrix = tf.confusion_matrix(ground_truth_input, predicted_indices,
                                                        num_classes=self.model_settings['label_count'])
        return predicted_indices,correct_prediction,confusion_matrix

