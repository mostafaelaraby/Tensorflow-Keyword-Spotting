# TensorFlow Keyword Spotting 
This project used to train a model which will be used to spot a set of specific keywords from input classes list,
it used several techniques like wav data augmentation (time shift addition of background noise , speed and stretching of input frequency) with several type of models like baseline Conv , VGGNET  and CTC model

### Prerequisites
You will need python 2.7/3.0 and TensorFlow 1.4

## Getting Started
You will need to write a yaml configuration file same as the configs for different models in example_config folder.

##Parameters 

###general_params
- seed used to randomize input data and input batches also used in the augmentation (randomly select the augmentation technique for each wav and its value)
- unknown_percentage percentage of unknowns in the training set (keywords that are not in the classes set)
- silence_percentage percentage of wavs not having a keyword to be spotted just background noise
- validation_percentage percentage of validation set
- testing_percentage percentage of test set in case of not using a testing list and having mode = 'test'

###Paths
used to specify data paths same format of [Speech Commands dataset](http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz) each folder containing a set of wav and the parent folder name specify the keyword ,
and needs to specify model path tmp directory for logs and the test set path that needs prediction and the background noise folder name.

###classes
the keywords that needs to be spotted from the provided dataset any other keywords available in the dataset will be treated as unknown

###wav\_reading_params
used to specify sampling rate (of input wav files ), time shift in millisecond , training clips duration in milliseconds , window size milliseconds , window stride  milliseconds and finger print type (mfcc,mel and log_mel) and ctc flag to denote using of ctc or not.

###model
- used to specify model used 
Baseline   as in [ http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf]( http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf "Convolutional Neural Networks for Small-footprint Keyword Spotting") - 
- VGGNET workigng on 2D fingerprints 
- LSTM for CTC models

###model_params
- loss could be crossentropy or ctc for ctc models
- Optimizedr can be SGD , Adam , Adagrad and Momentum
- training_steps number of training list as string 1000,2000
- learning_rate list of learning rate at each starting step from training_step list corresponding for example 0.01,0.001 will train till step 1K with learning 0.01 and  from step 1K to 3K using 0.001 learning rate
- save_eval_step_interval interval of train steps to save a checkpoint and evaluate
- dropout keep probability dropout
- batch_size size of training batch
- rnd_mini_batches True to use random batches as the one used in [https://github.com/tensorflow/tensorflow/blob/57b32eabca4597241120cb4aba8308a431853c30/tensorflow/examples/speech_commands/input_data.py#L398](https://github.com/tensorflow/tensorflow/blob/57b32eabca4597241120cb4aba8308a431853c30/tensorflow/examples/speech_commands/input_data.py#L398 "Tensorflow speech tutorial")
False to ensure iterating over the whole dataset

###augmentation
- Ops : operations to be used speed and stretch
- percentage of augmentation for each available class

###mode
test to test your models and train to use all data

## Results 
using speech commands test set data
- baseline.yml config  test accuracy 90.4% 
- vggnet.yml config test accuracy 92.3$ 

##TODO 
- Language model for CTC
- Add Resnet model

##References
- CTC Tensorflow [https://github.com/philipperemy/tensorflow-ctc-speech-recognition](https://github.com/philipperemy/tensorflow-ctc-speech-recognition "CTC tensorflow")
 
- Speech Recognition Tutorial by tensorflow [https://www.tensorflow.org/versions/master/tutorials/audio_recognition](https://www.tensorflow.org/versions/master/tutorials/audio_recognition "Simple Speech Recognition tensorflow tutorial")




