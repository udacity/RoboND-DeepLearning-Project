[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

## Deep Learning Project ##

In this project, you will train a deep neural network to identify and track a target in simulation and then issue commands to a drone to follow that target. So-called “follow me” applications like this are key to many fields of robotics and the very same techniques you apply here could be extended to scenarios like advanced cruise control in autonomous vehicles or human-robot collaboration in industry.

[image_0]: ./docs/misc/sim_screenshot.png
![alt text][image_0] 

## Setup Instructions
**Clone the repository**
```
$ git clone https://github.com/udacity/RoboND-DeepLearning.git
```

**Download the data**

Save the following two files into the data folder of the cloned repository. 

[Training Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/train.zip) 

[Validation Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/validation.zip)

**Download the QuadSim binary**

To interface your neural net with the QuadSim simulator, you must use a version QuadSim that has been custom tailored for this project. The previous version that you might have used for the Controls lab will not work.

The simulator binary can be downloaded [here](https://github.com/udacity/RoboND-DeepLearning/releases/latest)

**Install Dependencies**

You'll need Python 3 and Jupyter Notebooks installed to do this project.  The best way to get setup with these if you are not already is to use Anaconda following along with the [RoboND-Python-Starterkit](https://github.com/ryan-keenan/RoboND-Python-Starterkit).

If for some reason you choose not to use Anaconda, you must install the following frameworks and packages on your system:
* Python 3.x
* Tensorflow 1.2.1
* NumPy 1.11
* OpenCV 2
* SciPy 0.17.0
* eventlet 
* Flask
* h5py
* PIL
* python-socketio
* scikit-image
* socketIO-client
* transforms3d

## Implement the Segmentation Network
1. Download the training dataset from above and extract to the project `data` directory.
2. Complete `make_model.py`by following the TODOs in `make_model_template.py`
3. Complete `data_iterator.py` by following the TODOs in `data_iterator_template.py`
4. Complete `train.py` by following the TODOs in `train_template.py`
5. Train the network locally, or on [AWS](docs/aws_setup.md).
6. Continue to experiment with the training data and network until you attain the score you desire.
7. Once you are comfortable with performance on the training dataset, see how it performs in live simulation!

## Collecting Training Data ##
A simple training dataset has been provided above in this repository. This dataset will allow you to verify that you're segmentation network is semi-functional. However, if you're interested in improving your score, you may be interested in collecting additional training data. To do, please see the following steps.

The data directory is organized as  follows:
```
data/runs - contains the results of prediction runs
data/train/images - contains images for the training set
data/train/masks - contains masked (labeled) images for the training set
data/validation/images - contains images for the validation set
data/validation/masks - contains masked (labeled) images for the validation set
data/weights - contains trained TensorFlow models
```

### Training Set: with Hero Present ###
1. Run QuadSim
2. Select `Use Hero Target`
3. Select `With Other Poeple`
4. Click the `DL Training` button
5. With the simulator running, press "r" to begin recording.
6. In the file selection menu navigate to the `data/train/target/run1` directory
7. **optional** to speed up data collection, press "9" (1-9 will slow down collection speed)
8. When you have finished collecting data, hit "r" to stop recording.
9. To exit the simulator, hit "`<esc>`"

### Training Set: without Hero Present ###
1. Run QuadSim
2. Make sure `Use Hero Target` is **NOT** selected
3. Select `With Other Poeple`
4. Click the `DL Training` button
5. With the simulator running, press "r" to begin recording.
6. In the file selection menu navigate to the `data/train/non_target/run1` directory.
7. **optional** to speed up data collection, press "9"  (1-9 will slow down collection speed)
8. When you have finished collecting data, hit "r" to stop recording.
9. To exit the simulator, hit "`<esc>`"

### Validation Set ###
To collect the validation set, repeat both sets of steps above, except using the directory `data/validation` instead rather than `data/train`.

### Image Preprocessing ###
Before the network is trained, the images first need to be undergo a preprocessing step.
**TODO**: Explain what preprocessing does, approximately.
To run preprocesing:
```
$ python preprocess_ims.py
```
**Note**: If your data is stored as suggested in the steps above, this script should run without error.

## Training, Predicting and Scoring ##
With your training and validation data having been generated or downloaded from the above section of this repository, you are free to begin working with the neural net.

**Note**: Training CNNs is a very compute-intensive process. If your system does not have a recent Nvidia graphics card, with [cuDNN](https://developer.nvidia.com/cudnn) installed , you may need to perform the training step in the cloud. Instructions for using AWS to train your network in the cloud may be found [here](docs/aws_setup.md)

### Training your Model ###
**Prerequisites**
- Net has been implemented as per these instructions
- Training data is in `data` directory

To train, simply run the training script, `train.py`, giving it the name of the model weights file as a parameter:
```
$ python train.py my_amazing_model.h5
```
After the training run has completed, your model will be stored in the `data/weights` directory as an [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) file.

### Predicting on the Validation Set and Evaluating the Results ###
**Prerequisites**
-Model has been trained
-Validation set has been collected

Once the network has been trained, you can run inference on the validation set using `predict.py`. This script requires two parameters, the name of the model file you wish to perform prediction with, and the output directory where you would like to store your prediction results.

```
$ python predict.py my_amazing_model.h5 my_prediction_run
```

For the prediction run above, the results will be stored in `data/runs/my_prediction_run`.

To get a sense of the overall performance of the your net on the prediction set, you can use `evaluation.py` as follows:

```
$ python evaluate.py validation my_prediction_run
average intersection over union 0.34498680536
number of validation samples evaluated on 1000
number of images with target detected: 541
number of images false positives is: 4
average squared pixel distance error 11.0021170157
average squared log pixel distance error 1.4663195103
```

## Scoring ##
**TODO**

**How the Final score is Calculated**

**TODO**

**Ideas for Improving your Score**

**TODO**

**Obtaining a leaderboard score**

**TODO**

## Experimentation: Testing in Simulation
1. Copy your saved model to the weights directory `data/weights`.
2. Launch the simulator, select "Spawn People", and then click the "Follow Me" button. 
3. Run `server.py` to launch the socketio server.
4. Run the realtime follower script `$ realtime_follower.py my_awesome_model.h5`

**Note:** If you'd like to see an overlay of the detected region on each camera frame from the drone, simply pass the `--overlay_viz` parameter to `realtime_follower.py`
