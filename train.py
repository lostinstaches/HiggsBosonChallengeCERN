#!/usr/bin/env python

from utils import dataset_cleaning

TRAIN_DATASET = 'data/train.csv'


if __name__ == '__main__':
    metadata = dataset_cleaning.load_train_dataset(TRAIN_DATASET)
    print(metadata)
