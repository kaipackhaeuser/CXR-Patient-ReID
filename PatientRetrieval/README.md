# Patient re-identification
This repository provides code corresponding to the patient re-identification approach proposed in our paper "Deep 
learning-based patient re-identification is able to exploit the biometric nature of medical chest X-ray data". Using 
this approach, the patient re-identification capabilities of a deep learning architecture have been investigated. We 
used techniques from the field of content-based image retrieval. For training, validation and testing, the public NIH 
ChestX-ray14 dataset was considered. Moreover, some of our trained models were evaluated on the CheXpert dataset and the 
COVID-19 Image Data Collection.


## Data
The folder `./csv_files/` contains the csv files for training, validating and testing on the official split of the 
ChestX-ray14 dataset (https://nihcc.app.box.com/v/ChestXray-NIHCC), as well as for testing on the CheXpert dataset 
(https://stanfordmlgroup.github.io/competitions/chexpert) and the Covid-19 Image Data Collection 
(https://github.com/ieee8023/covid-chestxray-dataset). Note that the ChestX-ray14 dataset has been split into patients 
with only one image and patients with multiple images. For the training run from the paper, only the patients with 
multiple images were used. In contrast, all available patients were used for validation and testing. Still, for the sake 
of completeness, the csv file containing the single images from the training set has also been included in the folder.

Note that the respective datasets must be downloaded before running our programs. Please make sure to specify the 
correct image paths in `config.json`.

## System requirements
Training was done using an NVIDIA V100 with 32 GB of GPU memory. It took around 72 hours to fully train the network with 
4 workers and the given parameters in this setting.
We used Python 3.8.5 and performed our experiments on a Linux System running Ubuntu 18.04.5 LTS.

## Installation guide
To install some required dependencies, we provide a requirements.txt containing all required packages. Simply run the 
following pip3 command:
```
pip3 install -r requirements.txt
```
Once the Python packages have been installed, our program is ready to use. 

## Programm execution
To reproduce our experiments, the scripts should be executed in the following order:
```
python3 paper_retrieval_training_phase1.py --config_path './config_files/' --config 'config.json'
python3 paper_retrieval_training_phase2.py --config_path './config_files/' --config 'config.json'
```
The model created by the latter script can then be used in the testing scripts, invoked by calling: 
```
python3 paper_retrieval_test_resolution.py --config_path './config_files/' --config 'config.json'
python3 paper_retrieval_test_covid_data.py --config_path './config_files/' --config 'config.json'
python3 paper_retrieval_test_chexpert_data.py --config_path './config_files/' --config 'config.json'
```
Note that we provide a config file (`config.json`) which is needed for all experiments. Prior to code execution, this 
file should be adapted to define all the necessary hyperparameters and the correct image folder paths.

## Additional Information
The dataloader used for training is to a certain degree sensitive with respect to the batch size, due to this work's 
nature of using an online mining approach. Therefore, using a batch size smaller than 32 might lead to unexpected 
results. Another important aspect is the use of OneCycle training and its dependency on setting the amount of training 
epochs/iterations beforehand. Therefore, when using a different training set than the one provided by us, one 
would have to change the amount of iterations in the `configure_optimizers()` method in class complete_Model. 
The passage has been marked in `paper_retrieval_training_phase1.py` and `paper_retrieval_training_phase2.py` with a 
comment.

## Reproducibility
We provide our best-performing re-identification model in folder `./models/`. To reproduce our results without training 
the network from scratch, simply run one of our python scripts for testing (either `paper_retrieval_test_resolution.py`, 
`paper_retrieval_test_covid_data.py`, or `paper_retrieval_test_chexpert_data.py`).

To investigate whether foreign material in the images affects the verification performance, we evaluated our network on 
two small manually created subsets. To replicate these experiments, simply run the following two commands:
```
python3 paper_retrieval_test_subset_with_foreign_material.py --config_path './config_files/' --config 'config.json'
python3 paper_retrieval_test_subset_without_foreign_material.py --config_path './config_files/' --config 'config.json'
```
Note that you should download the respective datasets in advance (explained above) in order to be able to run these 
files. Also note to set the correct `final_model_name` in `config.json`. Our provided model is called
`retrieval_approach_final_model.pth`.
