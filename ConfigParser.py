import yaml
import numpy as np
import random
import utils


class ConfigParser:
    def __init__(self,config_path):
        self.max_num_wavs_per_class = 2 ** 27 - 1
        self.background_noise = '_background_noise_'
        self.unknown_label = 'unknown'
        self.silence_label = 'silence'
        self.seed = 0
        self.unknown_percentage = 10  # means 10%
        self.silence_percentage = 10  # means 10%
        self.testing_percentage = 10  # means 10%
        self.validation_percentage = 10  # means 10%
        self.model_path = ''
        self.data_path = ''
        self.test_data_names = {}
        self.val_data_names = {}
        self.tmp_dir = ''
        self.predict_path = ''
        self.classes = []
        self.num_classes = len(self.classes) + 2
        # params of audio processor
        self.time_shift_ms = 100
        self.sample_rate = 16000
        self.clip_duration_ms = 1000
        self.window_size_ms = 30
        self.window_stride_ms = 10
        self.dct_coefficient_count = 40
        self.with_ctc = False
        self.fingerprint_type = 'mfcc'  # mel , mfcc or log_mel
        self.model_name = 'baseline'  # baseline is conv model , sanity model for sanity checking of the  full pipeline , vggnet conv model  and lstm used for ctc models
        self.loss = 'crossentropy'  # or ctc loss for ctc models
        self.optimizer = 'SGD'  # SGD, Adam Momentum ,Adagrad
        self.training_steps = '15000,3000'
        self.save_eval_step_interval = 1000
        self.learning_rate = '0.01,0.001'
        self.dropout = 0.8  # 0.6 for baseline
        self.batch_size = 128
        self.num_hidden = 100
        self.num_layers = 1
        self.background_volume = 0.1  # volume of background noise
        self.background_frequency = 0.8  # background noise percentage
        self.augmentation_ops = []
        self.augmentation_percentage = 0  # % of each class wil be augmented
        self.mode = 'test'
        self.config_data = []
        self.rnd_mini_batches = True #used for totally randomized mini batches doesn't ensure passing over all dataset
        self.parse_config(config_path)
        self.model_settings = utils.prepare_model_settings( self.num_classes, self.sample_rate, self.clip_duration_ms, self.window_size_ms,self.window_stride_ms, self.dct_coefficient_count,self.time_shift_ms,self.max_num_wavs_per_class)
        random.seed(self.seed)
        np.random.seed(self.seed)

    def extract_var_val(self,parent_name):
        parent_node = self.config_data[parent_name]
        for param_name in parent_node:
            if isinstance(parent_node[param_name] ,str):
                exec('self.' + param_name + ' = \'' + str(parent_node[param_name]).strip() + '\'')
            else:
                exec('self.' + param_name + ' = ' + str(parent_node[param_name]).strip())

    def parse_config(self,config_path):
        self.config_data = yaml.load(open(config_path, 'r'))
        self.model_name = self.config_data['model']
        self.extract_var_val('general_params')
        self.extract_var_val('paths')
        self.classes = self.config_data['classes']
        self.num_classes = len(self.classes) + 2
        self.extract_var_val('wav_reading_params')
        self.extract_var_val('model_params')
        self.augmentation_ops = self.config_data['augmentation']['ops']
        self.augmentation_percentage = self.config_data['augmentation']['percentage']
        self.mode = self.config_data['mode']
        if 'validation_list' in self.config_data:
            lines = open(self.config_data['validation_list'], 'r').readlines()
            for line in lines:
                self.test_data_names[line.strip()] = 0
        if 'testing_list' in self.config_data:
            lines = open(self.config_data['testing_list'], 'r').readlines()
            for line in lines:
                self.val_data_names[line.strip()] = 0







