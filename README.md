This repo holds a collection of scripts trying to solve the problem from an old Kaggle competition - [distracted driver detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection/data). 

#### Getting started

To get started, you have to:
 1. Download the competition files into the `data` directory.
 2. Unzip the files:
 ```bash
unzip driver_imgs_list.csv.zip
unzip imgs.zip
```
3. Run the `validation_split.py` script. This script splits the training set into validation set and training set proper based on the identity of the driver. It also creates a tiny sampled versions of train, validation and test sets for testing.

#### Running experiments
To try a new combination of neural network architecture + network parameters + data augmentation parameters run `train.py`.  This script trains the model, creates a submission and prints the best loss and accuracy. The model is trained with early stopping based on validation loss. Best version of the model is saved. If you run the same experiment the second time, it will just print the saved best loss and accuracy and skip model training and creating the submission.

You can run the experiment like this:
```bash
model=m1
target_width=240
target_height=180
batch_size=64
width_shift_range=0.15
height_shift_range=0.15
rotation_range=30
shear_range=20
zoom_range=0.1
fill_mode=nearest
model_additional_params='19 32'
python train.py $model $target_width $target_height $batch_size $width_shift_range $height_shift_range $rotation_range $shear_range $zoom_range $fill_mode $model_additional_params
```
or simply
```bash
python train.py m1 240 180 64 0.15 0.15 30 20 0.1 nearest 19 32
```

Model weights, history and submission will be saved in `output/m1_vgg16/240_180_19_32/hsr:0.15,wsr:0.15,rr:30,th:180,sr:20.0,tw:240,fm:nearest,zr:0.1`
(the path is determined by model parameters).

You can also run the model with `--test` flag. In test run, all the same steps are followed - training, validation, creating submission, but now they run on a tiny sampled versions of train-, validation- and test- sets. Model weights and submission land in `test_runs` instead of `output` directory.

It's convenient to put many different experiments in a bash script - see `run.sh`. Even if you have to kill it halfway through, all the results are saved, so it will start where it left off, the next time you run it. 

To test a new neural architecture you need to create a model wrapper class with `get_model` and `path` methods (see e.g. `m1_vgg16.py`)and import it in the `train.py`.
