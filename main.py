"""
Domain-Aware Data selection for Speech Classification via Meta-Reweighting

This software is free of charge under research purposes.
For commercial purposes, please contact the authors.

-------------------------------------------------------------------------
File: main.py
 - train the model with speech data frommulti-source domains 
"""


import os
import pandas as pd
import numpy as np
import pickle
import copy
import torch
from torch import nn
import torch.nn.functional as F
import argparse
import random
from DOREME import doreme
from collections import Counter
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="sti")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--epoch", type=int, default=1000)

def read_pickle(filename):
    """ Read data from pickle file
    
    Args:
        filename (str): path to pickle file

    Returns:
        [numpy.array, numpy.array]: [data, target]
    """
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

def split_data(data, test_size):
    """ split data based on the given test size
    
    Args:
        data ([numpy.array, numpy.array]): data to split
        test_size (float): proportion of the test 

    Returns:
        [numpy.array, numpy.array], [numpy.array, numpy.array]: train data, test data
    """
    train_x, test_x, train_y,test_y = train_test_split(data[0], data[1], test_size=test_size, shuffle=True, stratify=data[1])
    return [train_x, train_y], [test_x, test_y]

def load_data(output_path, target_lang, test_size):
    """ load data to train
    
    Args:
        output_path (str): path to the file
        target_lang (str):target language to set as target domain
        test_size (float): proportion of the test
    
    Returns:
        [numpy.array, numpy.array], list, int, : train data, test data, [numpy.array, numpy.array], [numpy.array, numpy.array]: source domain data, names of domains, number of class, train data of the target domain, test data of the target domain
    """
    source_domains = []
    domain_names = []
    n_classes = []
    for lang in langs:
        if lang != target_lang:
            dataset = read_pickle(output_path+lang)
            source_domains.append(dataset)
            domain_names.append(lang)
        else:
            target_domain = read_pickle(output_path+lang)
        n_classes.append(len(Counter(dataset[1])))
    target_domain_train, target_domain_test = split_data(target_domain, test_size=test_size)
    return source_domains, domain_names, max(n_classes), target_domain_train, target_domain_test

if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.init()
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    if args.data == "zenodo":
        output_path = "zenodo/"
        langs = os.listdir(output_path)
        data_name = "zenodo"
        target_lang = "marche.pkl"
        test_size = 0.9
        source_domains, domain_names, n_classes, target_domain_train, target_domain_test = load_data(output_path, target_lang, test_size)

    elif args.data == "sti":
        output_path = "speech-to-intent/"
        langs = os.listdir(output_path)
        target_lang = "Punjabi.pkl"
        data_name = "sti"
        test_size = 0.9
        source_domains, domain_names, n_classes, target_domain_train, target_domain_test = load_data(output_path, target_lang, test_size)
    
    model = doreme(
        gpu = args.gpu,
        seed = args.seed,
        num_classes = n_classes,
    )

    model.fit(
        target_domain_train,
        source_domains,
        target_domain_test,
        epochs=args.epoch,
    )
