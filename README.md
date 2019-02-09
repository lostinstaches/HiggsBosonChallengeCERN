# EPFL Higgs Boson Machine Learning Challenge

### Overview
We present our machine learning system and approach used for solving Higgs Boson Machine Learning Challenge. Several methods for training predictive model are evaluated. To improve accuracy, data is preprocessed accordingly
before inputting it into algorithm of choice. We run several
experiments to pick hyperparameters. Final accuracy achieved is 0.804.

### Introduction

In 2014 kaggle launched Higgs Boson Machine Learning
Challenge. The competition explored possibility of using machine learning methods in particle physics research. As part of
the contest, the dataset containing descriptions of thousands
of particle collision events was published. Each event was
labelled either as signal or as background, depending on
whether the event may contain data leading to discovery of
new particles (signal) or not (background). The goal of the
challenge was to label the unlabeled events.
This report describes our approach used for solving the
challenge, adapted for needs of EPFL’s Machine Learning
course project. First, we briefly explore dataset. Then we describe all machine learning algorithms considered for solving
the problem and explain our reasoning behind our algorithm
of choice: logistic regression. We also provide high level
descriptions of our machine learning system. Finally, we
describe experiments that led us to appropriate hyperparameter
choice.

### Dataset
The dataset contains 30 learnable features and 250000
training examples. The examples are labeled by ’s’, if they are
positive example, and ’b’ if they are negative example. The
problem is therefore a straightforward classification problem
- we want to predict a label for each example based on all 30
features available.
