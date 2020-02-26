# Graph Neural Solver applied to a Linear System

DISCLAIMER : I am currently modifying and refactoring this repo, so there are many things that won't work perfectly.


This repo provides some code for:
* Building a dataset of random linear systems
* Building and training a Graph Neural Solver that solves these linear systems
* Reloading a previous model, and perform inference
* A visualization notebook

The Graph Neural Solver algorithm has been introduced in [Graph Neural Solver for Power Systems](https://hal.archives-ouvertes.fr/hal-02175989v1) and [Neural Networks for Power Flow : Graph Neural Solver](https://hal.archives-ouvertes.fr/hal-02372741v1).
It relies on Graph Neural Networks. More info about this work can be found [here](https://bdonon.github.io/gns_linear_system_article.html).

## Installation

Firstly, I recommend that you create a virtual environment. (You need to have virtualenv installed for this step, see https://virtualenv.pypa.io/en/latest/)
~~~~
cd ./GraphNeuralSolver_LinearSystem/
virtualenv -p python3 ENV
~~~~
Then you can activate it
~~~~
source ENV/bin/activate
~~~~
All the instructions below assume that the virtual environment is activated. 

Now install the requirements. 
*If you have a GPU (which you have setup to work smoothly with Tensorflow 1.14), then use requirements-gpu.txt instead of requirements.txt*
~~~~
pip install -r requirements.txt
~~~~

## Building the dataset

Instead of providing you with a dataset, you can find in data/ the piece of code I used to generate a set of random linear systems.
To generate a dataset:
~~~~
cd ./data
python build_dataset.py
~~~~
If you wish to modify the generated data, you can modify the parameters N_NODES, N_EDGES, etc.

Now go back to the previous directory
~~~~
cd ..
~~~~

## Training a model

Here is a typical command to start a training
~~~~
python main.py --max_iter=1000 --minibatch_size=100
~~~~
If you wish to have more informations about some args, try:
~~~~
python main.py --help
~~~~

## Monitoring a training process

First of all, there is a logger that keeps track of the training and validation losses.
Moreover you can use tensorboard to visualize the loss of the different layers of the Graph Neural Solver.
~~~~
tensorboard --logdir=./
~~~~

## Hyperparameter search

To perform a grid search, with for instance a learning rate in [1e-3, 1e-2], and a latent dimension in [5, 10]:
~~~~
python grid_search.py --learning_rate 1e-3 1e-2 --latent_dimension 5 10
~~~~
Once again, more info about all args are available using:
~~~~
python grid_search.py --help
~~~~

## Reloading a trained model

There is a trained model provided with the code : you just need to unzip the file that keeps track of the training loss:
~~~~
unzip results/1575991681/train.zip
~~~~

## Visualization notebook

Take a look at the visualization.ipynb notebook.
~~~~
ENV/bin/jupyter notebook
~~~~

# Infos and Contact
*Check out my blog bdonon.github.io for more info. If you have any questions or observations, contact me at balthazar.donon@rte-france.com*
