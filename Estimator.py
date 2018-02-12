import tensorflow as tf
from AudioReader import AudioReader
import numpy as np
import pandas as pd
import os
from tensorflow.python.framework import ops


class Estimator:
    def __init__(self, model_name='baseline', model=None, training_steps_list=[5], eval_save_every_step=1,
                 data_path='Train', tmp_dir='tmp', model_path='model',
                 background_frequency=0.8,background_volume=0.1,background_noise='',
                 learning_rates_list=[0.01],
                 batch_size=64,
                 dropout=0.4,
                 model_settings=None,
                 mode='test',
                 with_ctc=False,random_samples_mini_batch = False,
                 silence_label='silence',unknown_label='unknown',classes=[],
                 augmentation_ops=[], augmentation_percentage=0 ,
                 validation_percentage=10, testing_percentage=10,unknown_percentage = 10, silence_percentage=10,fingerprint_type='mfcc',testing_list={},validation_list={}):
        if len(training_steps_list) != len(learning_rates_list):
            raise Exception(
                '--how_many_training_steps and --learning_rate must be equal length '
                'lists, but are %d and %d long instead' % (len(training_steps_list),
                                                           len(learning_rates_list)))
        tf.logging.set_verbosity(tf.logging.INFO)
        ops.reset_default_graph() # used to reset default graph used by interactive session Interactive Session installs itself as the default session on construction
        self.data_path = data_path
        self.tmp_dir = tmp_dir
        self.model_path = model_path
        self.background_frequency = background_frequency
        self.background_volume = background_volume
        self.with_ctc = with_ctc
        self.random_samples_mini_batch = random_samples_mini_batch
        self.model_settings = model_settings
        self.model_name = model_name
        self.mode = mode
        self.audio_processor = AudioReader(audio_dir= self.data_path,model_settings= self.model_settings,silence_label= silence_label,unknown_label= unknown_label,classes= classes,background_noise = background_noise,augmentation_ops = augmentation_ops,augmentation_percentage = augmentation_percentage,
                                           fingerprint_type=fingerprint_type, mode=self.mode,validation_percentage=validation_percentage,testing_percentage=testing_percentage,
                                           unknown_percentage=unknown_percentage,silence_percentage= silence_percentage,testing_list = testing_list,validation_list= validation_list)
        self.silence_label = silence_label
        self.unknown_label = unknown_label
        self.classes = classes
        self.fingerprint_type = fingerprint_type
        self.batch_size = batch_size
        self.dropout = dropout
        self.sess = tf.InteractiveSession()
        # convert training_steps_list to training_steps
        self.training_steps_list = training_steps_list
        self.training_steps_max = np.sum(self.training_steps_list)
        self.learning_rates_list = learning_rates_list
        self.time_shift_samples = self.model_settings['time_shift_samples']
        self.eval_every_n_steps = eval_save_every_step
        self.features_2d = model_name=='vggnet' or model_name=='lstm'
        print(
            'Total Number of steps {} and eval every {} steps'.format(self.training_steps_max, self.eval_every_n_steps))
        print(
            'Total Number of Audio wavs {}'.format(self.audio_processor.set_size('training') ))
        self.fingerprint_input,self.ground_truth_input,self.seq_len = model.get_in_ground_truth()
        self.logits, self.dropout_prob = model.get_logits_dropout(self.fingerprint_input,self.seq_len)
        with tf.name_scope('Loss'):
            self.loss_mean = model.get_loss(self.logits,self.ground_truth_input,self.seq_len)
        tf.summary.scalar('Loss', self.loss_mean)
        with tf.name_scope('train'):
            self.learning_rate_input = tf.placeholder(tf.float32, [], name='learning_rate_input')
            self.train_step = model.get_optimizer(self.learning_rate_input, self.loss_mean)
        self.predicted_indices, self.correct_prediction, self.confusion_matrix = model.get_confusion_matrix_correct_labels(self.ground_truth_input,self.logits,self.seq_len,self.audio_processor)
        self.evaluation_step = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', self.evaluation_step)
        self.global_step = tf.train.get_or_create_global_step()
        self.increment_global_step = tf.assign(self.global_step, self.global_step + 1)
        self.saver = tf.train.Saver(tf.global_variables())
        # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
        self.merged_summaries = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.tmp_dir + '/train', self.sess.graph)
        self.validation_writer = tf.summary.FileWriter(self.tmp_dir + '/validation')
        tf.global_variables_initializer().run()
        self.start_step = 1
        tf.logging.info('Training from step: %d ', self.start_step)
        tf.train.write_graph(self.sess.graph_def, self.model_path, model_name + '.pbtxt')
        self.model  = model

    def load(self, checkpoint_path):
        self.saver.restore(self.sess, checkpoint_path)

    def train(self):

        for training_step in range(self.start_step, self.training_steps_max + 1):
            training_steps_sum = 0
            for i in range(len(self.training_steps_list)):
                training_steps_sum += self.training_steps_list[i]
                if training_step <= training_steps_sum:
                    learning_rate_value = self.learning_rates_list[i]
                    break
            # getting data for training
            if self.random_samples_mini_batch:
                train_fingerprints, train_ground_truth, f_names = self.audio_processor.get_data_random(self.batch_size, 0,
                                                                                                self.background_frequency,
                                                                                                self.background_volume,
                                                                                                self.time_shift_samples,
                                                                                                'training', self.sess,self.features_2d)
            else:
                train_fingerprints, train_ground_truth, f_names = self.audio_processor.get_data(self.batch_size, 0,
                                                                                                self.background_frequency,
                                                                                                self.background_volume,
                                                                                                self.time_shift_samples,
                                                                                                'training', self.sess,
                                                                                                self.batch_size,
                                                                                                training_step,features_2d=self.features_2d)
            if self.with_ctc:
                train_inputs, train_targets, train_seq_len = self.model.convert_batch_to_ctc_format(train_fingerprints, f_names,self.audio_processor)
                train_accuracy = 0
                loss_value = 0
                for itm_index, train_itm in enumerate(train_inputs):
                    train_summary, accuracy, current_loss, _, _ = self.sess.run(
                        [
                            self.merged_summaries, self.evaluation_step, self.loss_mean, self.train_step,
                            self.increment_global_step
                        ],
                        feed_dict={
                            self.fingerprint_input: train_itm,
                            self.ground_truth_input: train_targets[itm_index],
                            self.learning_rate_input: learning_rate_value,
                            self.seq_len: train_seq_len[itm_index]
                        })
                    train_accuracy += accuracy * self.batch_size
                    loss_value += current_loss * self.batch_size
                    self.train_writer.add_summary(train_summary, training_step)
                train_accuracy /= self.audio_processor.set_size('training')
                loss_value /= self.audio_processor.set_size('training')
            else:
                # run the graph with this batch of training data.
                train_summary, train_accuracy, loss_value, _, _ = self.sess.run(
                    [
                        self.merged_summaries, self.evaluation_step, self.loss_mean, self.train_step,
                        self.increment_global_step
                    ],
                    feed_dict={
                        self.fingerprint_input: train_fingerprints,
                        self.ground_truth_input: train_ground_truth,
                        self.learning_rate_input: learning_rate_value,
                        self.dropout_prob: self.dropout
                    })
            self.train_writer.add_summary(train_summary, training_step)
            tf.logging.info('Step #%d: rate %f, accuracy %.1f%%, Loss %f' %
                            (training_step, learning_rate_value, train_accuracy * 100,
                             loss_value))
            is_last_step = (training_step == self.training_steps_max)
            if (training_step % self.eval_every_n_steps) == 0 or is_last_step:
                set_size = self.audio_processor.set_size('validation')
                total_accuracy = 0
                total_conf_matrix = None
                for i in range(0, set_size, self.batch_size):
                    validation_fingerprints, validation_ground_truth, f_names = (
                        self.audio_processor.get_data_random(self.batch_size, i , 0.0, 0.0, 0,
                                                             'validation', self.sess,self.features_2d))
                    batch_size_local = min(self.batch_size, set_size - i)
                    validation_summary, validation_accuracy, conf_matrix = self.eval(validation_fingerprints,
                                                                                     validation_ground_truth, f_names,
                                                                                     batch_size_local, set_size)
                    self.validation_writer.add_summary(validation_summary, training_step)

                    total_accuracy += (validation_accuracy * batch_size_local) / set_size
                    if total_conf_matrix is None:
                        total_conf_matrix = conf_matrix
                    else:
                        total_conf_matrix += conf_matrix
                    tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
                    tf.logging.info('Step %d: Validation accuracy = %.1f%% (N=%d)' % (
                        training_step, total_accuracy * 100, set_size))
                    checkpoint_path = os.path.join(self.model_path, self.model_name + '.ckpt')
                    tf.logging.info('Saving to "%s-%d"', checkpoint_path, training_step)
                    self.saver.save(self.sess, checkpoint_path, global_step=training_step)
        if self.mode == 'test':
            set_size = self.audio_processor.set_size('testing')
            tf.logging.info('set_size=%d', set_size)
            total_accuracy = 0
            total_conf_matrix = None
            for i in range(0, set_size, self.batch_size):
                test_fingerprints, test_ground_truth, f_names = self.audio_processor.get_data_random(self.batch_size, i,
                                                                                                     0.0, 0.0, 0,
                                                                                                     'testing',
                                                                                                     self.sess,self.features_2d)

                batch_size_local = min(self.batch_size, set_size - i)
                _, test_accuracy, conf_matrix = self.eval(test_fingerprints, test_ground_truth, f_names,
                                                          batch_size_local, set_size)

                if conf_matrix is not None:
                    total_accuracy += (test_accuracy * batch_size_local) / set_size
                    if total_conf_matrix is None:
                        total_conf_matrix = conf_matrix
                    else:
                        total_conf_matrix += conf_matrix
            tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
            tf.logging.info('Final test accuracy = %.1f%% (N=%d)' % (total_accuracy * 100, set_size))
        return total_conf_matrix, total_accuracy * 100, set_size

    def predict(self, test_data_path):
        test_data_reader = AudioReader(audio_dir= test_data_path,model_settings= self.model_settings,background_noise='', train=False,silence_label=self.silence_label,
                                       unknown_label = self.unknown_label,classes = self.classes,fingerprint_type = self.fingerprint_type)
        set_size = test_data_reader.set_size('testing')
        results = []
        index = []
        for i in range(0, set_size, self.batch_size):
            test_fingerprints, test_ground_truth, fnames = test_data_reader.get_data_random(self.batch_size, i,  0.0,
                                                                                            0.0, 0, 'testing',
                                                                                            self.sess,self.features_2d)
            if self.with_ctc:
                test_fingerprints, test_ground_truth, test_seq_len = self.model.convert_batch_to_ctc_format(test_fingerprints, fnames,self.audio_processor)
                test_preds = []
                for itm_index, eval_itm in enumerate(test_fingerprints):
                    pred = self.sess.run(
                        [self.predicted_indices],
                        feed_dict={
                            self.fingerprint_input: eval_itm,
                            self.ground_truth_input: test_ground_truth[itm_index],
                            self.seq_len: test_seq_len[itm_index]
                        })
                    test_preds.append(pred[0])
                test_preds = np.array(test_preds)
            else:
                test_preds = self.sess.run(
                    [self.predicted_indices],
                    feed_dict={
                        self.fingerprint_input: test_fingerprints,
                        self.ground_truth_input: test_ground_truth,
                        self.dropout_prob: 1.0
                    })
            test_preds = test_data_reader.label_to_names(test_preds)
            index.extend([os.path.basename(fname) for fname in fnames])
            results.extend(test_preds)
        df = pd.DataFrame(columns=['fname', 'label'])
        df['fname'] = index
        df['label'] = results
        return df

    def eval(self, eval_fingerprints, eval_ground_truth, f_names, batch_size_local, set_size):
        conf_matrix = None
        if self.with_ctc:
            eval_inputs, eval_targets, eval_seq_len = self.model.convert_batch_to_ctc_format(eval_fingerprints, f_names,self.audio_processor)
            eval_accuracy = 0
            for itm_index, eval_itm in enumerate(eval_inputs):
                eval_summary, accuracy, val_conf_matrix = self.sess.run(
                    [self.merged_summaries, self.evaluation_step, self.confusion_matrix],
                    feed_dict={
                        self.fingerprint_input: eval_itm,
                        self.ground_truth_input: eval_targets[itm_index],
                        self.seq_len: eval_seq_len[itm_index]
                    })
                eval_accuracy += accuracy * batch_size_local
                if conf_matrix is None:
                    conf_matrix = val_conf_matrix
                else:
                    conf_matrix += val_conf_matrix
            eval_accuracy /= set_size
        else:
            eval_summary, eval_accuracy, conf_matrix = self.sess.run(
                [self.merged_summaries, self.evaluation_step, self.confusion_matrix],
                feed_dict={
                    self.fingerprint_input: eval_fingerprints,
                    self.ground_truth_input: eval_ground_truth,
                    self.dropout_prob: 1.0
                })
        return eval_summary, eval_accuracy, conf_matrix
