# higgs_boson_project

This repository contains script that runs logistic_regression training,
which generates prediction csv reaching 0.80400 on private leaderboard
(this is a little lower than our best prediction submitted, since best
prediction used undeterministic training, and score lowered after setting
seed to 1.

To generate predictions, run `python run.py`.

The repository contains 3 code files:
`run.py` - prediction generation and training setup script
`dataset.py` - data loading and preprocessing functions
`implementations.py` - implementations of methods mentioned in PDF and wrappers on top of them
