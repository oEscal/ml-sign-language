# ML Sign Language

# Set the environment
 - Open a linux terminal and type:
 ```bash
 $ virtualenv venv
 $ source venv/bin/activate
 $ pip install -r requirements.txt
 ```

# Files
 - `data_treatment.py` - python script to merge all data on the original dataset and split into 3 datasets (train, validation and test)
 - `neural_networks.py` - neural networks algoritm's code (run and save the classifier into a binary file)
 - `neural_networks_interpretation.py` - neural networks graphs and latex tables
 - `classifiers.py` - wrappers for the used classifiers
 - `utils.py` - some code shared between other python scripts
 - `svm_lr.py` - SVM and LR algoritm's code (run and save the classifier into a binary file)
 - `run_nn_several_times.sh` - bash script to run multiple times the `neural_networks.py` script, rotating the params of the algorithm
