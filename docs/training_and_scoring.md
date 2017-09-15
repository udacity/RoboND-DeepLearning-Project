# Semantic Segmentation
### Introduction
This trains on a FCN segmentation network using vgg16 as a base. 


##### Setting up a dataset

First generate data by running the unity sim. On the menu screen of the make sure the there are checks in `Use Hero Target`, and `With Other People` then click the DL training button.

Once the sim is running press `r` to start recording. In the file selection menu navigate to `data/robo_train/target/hero_train1` and click `select`. To speed up training press the `9` 

Open another simulation instance unselect the option for `Use Hero Target`, DL training training, press `r`, to start recording and navigate to the directory `data/robo_train/non_target/run_train1`

To generate validation data, repeat the above steps except select folders in `data/robo_validation`

After data is generated while in the project directory run `python preprocess_ims.py`. If your data is stored as above no configuration of this file is required, otherwise the directories it looks for the images in needs to be changed. 

##### Training, and predicting

Training and predicting is accessed by running `python run.py` there are several configuration options which for now are in the main function of `run.py`

To train without loading a checkpoint, set `load_checkpoint_name = None`

To load a checkpoint and generate prediction outputs, set `save_checkpoint_name=None` and `load_checkpoint_name=your_checkpoint.ckpt` and set `predict = True`

##### Scoring a set of predictions

Edit the main function of `score_predictions.py`,  set the name of the run you want to score `prediction_dir`, and the name of the checkpoint file to use in `checkpoint_file`