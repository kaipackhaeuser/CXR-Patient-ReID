# Patient verification

## Overview
This repository provides code corresponding to the patient verification approach proposed in our paper "Deep learning-based patient re-identification is able to exploit the biometric nature of medical chest X-ray data". Using this approach, the patient verification capabilities of a deep learning architecture 
have been investigated. More precisely, a siamese neural network was utilized to determine whether two arbitrary chest 
radiographs can be recognized to belong to the same patient or not. For training, validation and testing, the public NIH 
ChestX-ray14 dataset was considered. Additionally, the CheXpert dataset and the COVID-19 Image Data Collection were used 
for further evaluations.

Our program provides code to train and evaluate our siamese neural network architecture. A detailed instruction 
regarding system requirements, installation and usage is given below.

## System requirements
### Hardware requirements
Our package requires a computer with enough RAM to support the in-memory operations. Moreover, an NVIDIA GPU with at 
least 8GB of GPU memory is needed for fast computations during training and testing.

### Software requirements
#### OS requirements
The package has been tested on *Linux* operating systems.

#### Python version
This package requires Python 3. It was tested using Python 3.8.5.

#### Python dependencies
```
matplotlib==3.3.3
numpy==1.19.5
Pillow==8.1.0
scikit-learn==0.24.0
torch==1.7.1+cu110
torchvision==0.8.2+cu110
```

## Installation guide
To install the above mentioned dependencies, we provide a requirements.txt containing  all required packages. Simply 
run the following pip3 command:
```
pip3 install -r requirements.txt
```
Once the Python packages have been installed, our program is ready to use.

## Data
Note that the NIH ChestX-ray14 dataset images are required for our program. You can simply download the dataset from https://nihcc.app.box.com/v/ChestXray-NIHCC. You can 
save the dataset to any folder on your machine. However, before running the main script, please make sure to specify 
the correct image path in `config.json`.

The COVID-19 Image Data Collection is available on GitHub at https://github.com/ieee8023/covid-chestxray-dataset. The 
CheXpert dataset can be requested at https://stanfordmlgroup.github.io/competitions/chexpert.



## Instructions for use
Our program is a composition of several Python files. The two most important files are `main.py` and `config.json`. The 
main file is required to start a specific experiment, whereas the config file is needed to set the experiment-related 
hyper-parameters and to define the image path which leads to the radiographs available in the NIH ChestX-ray14 dataset. 
Note that we provide an example config file in folder `./config_files/` which can be used as a reference. 

A multitude of different parameters can be defined in `config.json`, which are explained below:
* "experiment_description": define a description for the conducted experiment
* "resumption": false (set to true if you want to resume a previous experiment)
* "resumption_count": null (set to an integer value if you resume an experiment, e.g. 1 for the first resumption)  
* "previous_experiment": null (in case of a resumption: set the experiment description of the experiment to be resumed in order to make sure that the corresponding checkpoint can be loaded)  
* "image_path": "/path_to_the_dataset/ChestX-ray8/images/images/" (please define the image path)
* "siamese_architecture": the network architecture incorporated in the two siamese branches (we chose "ResNet-50" in our work)
* "data_handling": the used data handling technique (choose either "balanced" or "randomized") 
* "num_workers": 16
* "pin_memory": true
* "n_channels": the numer of input channels (we chose 3 in our work to ensure compatibility with the pre-trained ResNet-50)  
* "n_features": 128
* "image_size": the selected image size(original image size is 1024x1024, we chose 256x256 in our work)  
* "loss": the selected loss function (we chose "BCEWithLogitsLoss" in our work)  
* "optimizer": the selected optimization method (we chose "Adam" in our work)  
* "learning_rate": the learning rate used during optimization (we conducted experiments with LRs in the range of 0.001 to 0.0000001)  
* "batch_size": the batch size used for our experiments (we chose a batch size of 32)  
* "max_epochs": define a maximum number of epochs (set to 100 in our work)  
* "early_stopping": the patience for the early stopping criterion (set to 5 in our work) 
* "transform": the transformation which is applied to the data (we chose "image_net" for our experiments)  
* "n_samples_train": amount of image pairs for training (maximum is 792294)
* "n_samples_val": amount of image pairs for validation (set to 50000 for all experiments)
* "n_samples_test": amount of image pairs for testing (set to 100000 for all experiments)

Note that we also provide the image pairs for training, validation and testing in folder `./image_pairs/`.

To use our program for network training and evaluation, call the main script (`main.py`) with the following command:
```
python3 main.py --config_path ./config_files/ --config config.json
```

## Output
As soon as the main script has been started with a pre-defined config.json file, 
the training of our siamese neural network starts. After successful training, the evaluation/testing phase begins. Once 
the evaluation is complete, the results of the conducted experiment can be investigated in the respective subfolder 
in `./archive/`.

Our program creates several files that document the results, including:
* a file reporting commonly-used evaluation metrics, like AUC score, accuracy, precision, recall, and more,
* the ROC metrics, i.e. false positive rate and true positive rate at various threshold settings,
* the resulting ROC curve stored as a .png image,
* the predictions made by our trained network stored in a .txt file, and
* the bootstrap 95% confidence intervals.

Moreover, we save:
* the network which performs best on our validation set,
* a complete checkpoint, in case one would like to resume an experiment at a later point, 
* the training/validation loss curves as a .png image and the corresponding values which are stored in a .txt file. 

As we are working with large amounts of data, especially in the training phase, optimizing our siamese neural network 
can take a long time. Note that this mainly depends on the amount of training data and the learning rate (which can be 
defined in config.json). Typically, the experiments finish within one day. For some experiments, especially when the 
maximum amount of training data and a low learning rate are chosen, training takes several days.


## Reproducibility
We provide our best-performing verification model in folder `./trained_models/` and a simple python script 
(`evaluate_trained_model.py`) to reproduce our best results. Simply run `evaluate_trained_model.py` with variable 
`n_samples` set to 100000:

```
python3 evaluate_trained_model.py --image_path <path_to_the_dataset/ChestX-ray8/images/images/>
```

To investigate whether foreign material in the images affects the verification performance, we evaluated our network on 
two small manually created subsets. To replicate these experiments, simply run the python script 
`evaluate_foreign_material.py`:

```
python3 evaluate_foreign_material.py --image_path <path_to_the_dataset/ChestX-ray8/images/images/> --artifacts
python3 evaluate_foreign_material.py --image_path <path_to_the_dataset/ChestX-ray8/images/images/> --no-artifacts
```

To evaluate our verification model on the CheXpert dataset or the COVID-19 Image Data Collection, we provide the python 
scripts `evaluate_model_chexpert.py` and `evaluate_model_covid19.py`:

```
python3 evaluate_model_chexpert.py --image_path <path_to_the_chexpert_dataset>
python3 evaluate_model_covid19.py --image_path <path_to_the_covid_dataset>
```
