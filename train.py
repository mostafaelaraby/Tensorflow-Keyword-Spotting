import tensorflow as tf
from utils import truncate_dir
from Estimator import Estimator
from ConfigParser import ConfigParser
from utils import model_factory_to_object

Config = None


def main(_):
    truncate_dir(Config.model_path)
    truncate_dir(Config.tmp_dir)

    training_steps_list = list(map(int, Config.training_steps.split(',')))
    learning_rates_list = list(map(float, Config.learning_rate.split(',')))
    #todo check for the latest checkpoint for the model that needs training
    #update train step number to continue training
    model_obj = model_factory_to_object(Config.model_name)
    model = model_obj(optimizer=Config.optimizer, loss=Config.loss, model_settings=Config.model_settings, train=True,num_hidden=Config.num_hidden,num_layers=Config.num_layers)
    estimator = Estimator(model_name=Config.model_name, model=model,data_path=Config.data_path,model_path=Config.model_path,tmp_dir=Config.tmp_dir,
                          eval_save_every_step=Config.save_eval_step_interval,
                          training_steps_list=training_steps_list, learning_rates_list=learning_rates_list,
                          batch_size=Config.batch_size, dropout=Config.dropout, model_settings=Config.model_settings, mode=Config.mode ,
                          with_ctc=Config.with_ctc,random_samples_mini_batch = Config.rnd_mini_batches,silence_label = Config.silence_label,unknown_label = Config.unknown_label,classes = Config.classes,
                          augmentation_ops=Config.augmentation_ops, augmentation_percentage=Config.augmentation_percentage,
                          validation_percentage= Config.validation_percentage, testing_percentage= Config.testing_percentage, unknown_percentage= Config.unknown_percentage,silence_percentage=Config.silence_percentage,
                          fingerprint_type=Config.fingerprint_type,background_frequency=Config.background_frequency,background_volume=Config.background_volume,background_noise=Config.background_noise,
                          testing_list= Config.test_data_names,validation_list=Config.val_data_names)
    #estimator.load(os.path.join(Config.model_path,'lstm.ckpt-1000')) to load the trained model
    total_conf_matrix, total_accuracy, set_size = estimator.train()
    print('Confusion Matrix:\n %s' % (total_conf_matrix))
    print('Final test accuracy = %.1f%% (N=%d)' % (total_accuracy, set_size))
    df = estimator.predict(Config.predict_path)
    df.to_csv( 'sub.csv', index=False)
    df = estimator.predict(Config.predict_path)
    df.to_csv( 'sub.csv', index=False)

if __name__ == '__main__':
    Config = ConfigParser('example_config/ctc.yml')
    tf.app.run(main=main)