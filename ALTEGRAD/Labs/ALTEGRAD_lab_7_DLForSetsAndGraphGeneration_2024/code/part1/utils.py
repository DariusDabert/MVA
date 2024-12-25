"""
Learning on Sets and Graph Generative Models - ALTEGRAD - Nov 2024
"""

import numpy as np


def create_train_dataset():
    n_train = 100000
    max_train_card = 10

    ############## Task 1
    
    X_train = []
    y_train = []

    for i in range(n_train):
        M = np.random.randint(1, max_train_card+1)
        elements = np.random.randint(1, 11, M)
        set = np.zeros(10)
        for j in range(M):
            set[10-M+j] = elements[j]
        X_train.append(set)
        y_train.append(np.sum(elements))

    return X_train, y_train


def create_test_dataset():
    
    ############## Task 2
    
    X_test = []
    y_test = []

    for i in range(5, 105, 5):
        x_sets = []
        y_sets = []
        for j in range(10000):
            elements = np.random.randint(0, 11, i)
            set = np.zeros(i)
            for k in range(i):
                set[k] = elements[k]
            x_sets.append(set)
            y_sets.append(np.sum(elements))
        X_test.append(np.array(x_sets))
        y_test.append(np.array(y_sets))

    return X_test, y_test
